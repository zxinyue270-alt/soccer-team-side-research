/* gbf web: the page's script (docs/design/web.md).
 *
 * Plain JavaScript without any external code. Everything it shows comes from the gbf server
 * on this machine. The access token comes from the address (#t=...), which the browser never
 * sends to a server; the script sends it with every API call in the X-GBF-Token header.
 * Images and videos of a run are loaded with the cookie GET /api/session sets.
 * Every text from the server is inserted as text (never as HTML).
 */
"use strict";

(() => {
  const TOKEN_HEADER = "X-GBF-Token";
  const POLL_MS = 1500;
  const state = { token: "", session: null, timer: 0, generation: 0 };
  const $ = (id) => document.getElementById(id);

  // ------------------------------------------------------------------ small helpers

  function el(tag, attrs, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (key === "class") node.className = value;
      else if (key.startsWith("on") && typeof value === "function") {
        node.addEventListener(key.slice(2), value);
      } else node.setAttribute(key, value === true ? "" : String(value));
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return node;
  }

  function chip(text, kind) {
    return el("span", { class: "chip" + (kind ? " chip-" + kind : "") }, text);
  }

  function hashParams() {
    return new URLSearchParams(location.hash.slice(1));
  }

  function link(extra) {
    const params = new URLSearchParams();
    params.set("t", state.token);
    for (const [key, value] of Object.entries(extra || {})) {
      if (value !== undefined && value !== null && value !== "") params.set(key, String(value));
    }
    return "#" + params.toString();
  }

  function fileUrl(run, rel) {
    const parts = String(rel).split("/").map(encodeURIComponent).join("/");
    return "/runs/" + encodeURIComponent(run) + "/" + parts;
  }

  function failure(message, problems, status) {
    const error = new Error(message);
    error.problems = problems || [];
    error.status = status || 0;
    return error;
  }

  async function api(path, options) {
    const opts = options || {};
    const init = {
      method: opts.method || "GET",
      headers: { [TOKEN_HEADER]: state.token },
      cache: "no-store",
      credentials: "same-origin",
    };
    if (opts.json !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(opts.json);
    }
    let response;
    try {
      response = await fetch(path, init);
    } catch (error) {
      throw failure(
        "连不上本机的 gbf 服务：它可能已经停止了。请在终端重新运行 python -m gbf web，再用新打印的地址打开。"
      );
    }
    let data = null;
    try {
      data = await response.json();
    } catch (error) {
      data = null;
    }
    if (!response.ok) {
      const message = (data && data.error_zh) || "请求失败（HTTP " + response.status + "）。";
      throw failure(message, (data && data.problems_zh) || [], response.status);
    }
    return data;
  }

  function fmtTime(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return iso;
    return date.toLocaleString("zh-CN", { hour12: false });
  }

  function fpsText(fps) {
    if (!fps) return "—";
    const [num, den] = String(fps).split("/").map(Number);
    const value = den ? num / den : num;
    if (!Number.isFinite(value)) return String(fps);
    return (Number.isInteger(value) ? String(value) : value.toFixed(2)) + " fps";
  }

  function duration(seconds) {
    if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "—";
    const whole = Math.round(seconds);
    const minutes = Math.floor(whole / 60);
    return minutes ? minutes + " 分 " + (whole % 60) + " 秒" : whole + " 秒";
  }

  function sizeText(bytes) {
    if (typeof bytes !== "number") return "";
    if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
    if (bytes >= 1024) return Math.round(bytes / 1024) + " KB";
    return bytes + " B";
  }

  function signed(value) {
    return value > 0 ? "+" + value : String(value);
  }

  function current(generation) {
    return generation === state.generation;
  }

  function fill(node, ...children) {
    // replaceChildren would insert null as the text "null"
    const kept = children.flat(Infinity).filter((c) => c !== null && c !== undefined && c !== false);
    node.replaceChildren(...kept);
    return node;
  }

  function setView(...nodes) {
    fill($("view"), ...nodes);
    window.scrollTo(0, 0); // a new view starts at the top of the page
  }

  function errorBox(error) {
    const problems = error.problems || [];
    return el(
      "div",
      { class: "alert alert-bad", role: "alert" },
      el("p", null, error.message || String(error)),
      problems.length ? el("ul", null, problems.map((p) => el("li", null, p))) : null
    );
  }

  const RUN_STATUS = {
    passed: ["完成", "good"],
    failed: ["失败", "bad"],
    unreadable: ["无法读取", "bad"],
    unknown: ["未知", ""],
  };

  function runStatusChip(status) {
    const [text, kind] = RUN_STATUS[status] || [status || "未知", ""];
    return chip(text, kind);
  }

  function tagChips(tags, banner) {
    const nodes = [];
    if (banner) nodes.push(chip(banner, "banner"));
    for (const tag of tags || []) nodes.push(chip(tag, "tag"));
    return nodes.length ? el("span", { class: "chips" }, nodes) : "—";
  }

  // ------------------------------------------------------------------ layout

  function renderNav(active) {
    const items = [
      ["runs", "分析结果", {}],
      ["new", "新建分析", { view: "new" }],
    ];
    const links = items.map(([key, label, extra]) =>
      el(
        "a",
        {
          href: link(extra),
          class: "nav-link" + (key === active ? " active" : ""),
          "aria-current": key === active ? "page" : null,
        },
        label
      )
    );
    fill($("nav"), links);
    $("brand").setAttribute("href", link({}));
  }

  function renderPrivacy() {
    const text = state.session && state.session.privacy_zh;
    fill($("privacy"), ...(text ? [el("span", { class: "dot" }), el("span", null, text)] : []));
  }

  // ------------------------------------------------------------------ the full-clip player

  function dockSpace() {
    // while the player is open, the page ends with room for it, so that its last rows can
    // be scrolled clear of the player
    const dock = $("dock");
    const space = dock.hidden ? 0 : dock.offsetHeight + 32;
    document.documentElement.style.setProperty("--dock-space", space + "px");
  }

  function foldDock(folded) {
    const button = $("dock-fold");
    const label = folded ? "展开播放器" : "收起播放器";
    $("dock").classList.toggle("folded", folded);
    button.setAttribute("aria-expanded", folded ? "false" : "true");
    button.setAttribute("aria-label", label);
    button.title = label;
    button.textContent = folded ? "▴" : "▾";
    if (folded) $("dock-video").pause();
    dockSpace();
  }

  function closeDock() {
    const video = $("dock-video");
    video.pause();
    $("dock").hidden = true;
    dockSpace();
  }

  function dockPlay(run, path, seconds, label) {
    const dock = $("dock");
    const video = $("dock-video");
    const url = fileUrl(run, path);
    dock.hidden = false;
    foldDock(false);
    $("dock-label").textContent = label;
    const seek = () => {
      try {
        video.currentTime = seconds;
      } catch (error) {
        return;
      }
      const playing = video.play();
      if (playing && playing.catch) playing.catch(() => {});
    };
    if (video.dataset.src !== url) {
      video.dataset.src = url;
      video.addEventListener("loadedmetadata", seek, { once: true });
      video.src = url;
      video.load();
    } else if (video.readyState >= 1) {
      seek();
    } else {
      video.addEventListener("loadedmetadata", seek, { once: true });
    }
  }

  function playButton(run, full, frame, clock, seconds, compact) {
    if (!full || seconds === null || seconds === undefined) return null;
    const label = "全片逐帧标注 · 第 " + frame + " 帧（" + (clock || "—") + "）";
    return el(
      "button",
      {
        type: "button",
        class: "play",
        title: "在全片逐帧标注（原始画面）里从第 " + frame + " 帧开始播放",
        "aria-label": "从第 " + frame + " 帧播放全片逐帧标注",
        onclick: () => dockPlay(run, full.path, seconds, label),
      },
      compact ? "▶" : "▶ 从这一帧播放"
    );
  }

  // ------------------------------------------------------------------ run list

  async function viewRuns(generation) {
    const data = await api("/api/runs");
    if (!current(generation)) return;
    const head = el(
      "div",
      { class: "page-head" },
      el("h1", null, "分析结果"),
      el("span", { class: "muted small" }, "运行目录：", el("code", null, data.runs_dir))
    );
    if (!data.runs.length) {
      setView(
        head,
        el(
          "div",
          { class: "panel empty" },
          el("p", null, "运行目录里还没有分析结果（一次运行是一个带 run.json 的子文件夹）。"),
          el("a", { class: "btn btn-primary", href: link({ view: "new" }) }, "新建分析")
        )
      );
      return;
    }
    const rows = data.runs.map((run) =>
      el(
        "tr",
        null,
        el("td", null, el("a", { class: "run-link", href: link({ run: run.name }) }, run.name)),
        el("td", { class: "num" }, fmtTime(run.created_at)),
        el("td", null, runStatusChip(run.status)),
        el(
          "td",
          null,
          run.video || "—",
          run.frames ? el("div", { class: "muted small" }, run.frames + " 帧 · " + fpsText(run.fps)) : null
        ),
        el("td", { class: "num" }, run.highlight_count === null ? "—" : String(run.highlight_count)),
        el("td", null, tagChips(run.restrictions, run.banner_zh))
      )
    );
    const headers = ["名称", "创建时间", "状态", "视频", "高光", "限制标签"];
    setView(
      head,
      el(
        "div",
        { class: "panel tablewrap" },
        el(
          "table",
          { class: "table runs" },
          el("thead", null, el("tr", null, headers.map((h) => el("th", { scope: "col" }, h)))),
          el("tbody", null, rows)
        )
      )
    );
  }

  // ------------------------------------------------------------------ one run

  function factsGrid(detail) {
    const video = detail.run.video || {};
    const teams = detail.run.team_names || {};
    const facts = [
      ["视频", video.file_name || "—"],
      ["帧数", video.frames || detail.frame_count || "—"],
      ["帧率", fpsText(detail.fps || video.fps)],
      ["分辨率", video.width && video.height ? video.width + "×" + video.height : "—"],
      ["原声", detail.run.audio_source || "—"],
      ["高光", String(detail.highlights.length)],
      ["创建时间", fmtTime(detail.run.created_at)],
      [detail.automatic ? "处理总用时" : "用时", duration(detail.automatic ? detail.automatic.wall_s : detail.run.wall_s)],
    ];
    if (detail.automatic) facts.push(["其中后半段分析", duration(detail.run.wall_s)]);
    if (teams.left || teams.right) {
      facts.push(["球队", "左：" + (teams.left || "—") + " · 右：" + (teams.right || "—")]);
    }
    return el(
      "div",
      { class: "facts" },
      facts.map(([key, value]) =>
        el("div", { class: "fact" }, el("div", { class: "k" }, key), el("div", { class: "v" }, String(value)))
      )
    );
  }

  function honestyNotes(detail) {
    const items = [
      "所有数值都是程序自动估计的，误差还没有对照真值测量，所以一律标“未验证”。",
      "分析都在原始画面上做：关键时刻和测量都对应原视频的帧号。",
      "3D 骨架只是参考：来自它的数值都标“3D 参考”（误差测完、并且赢过 2D 方法之前只供参考），它的结果也画回原始画面上看。",
    ];
    if (detail.full_clip) {
      items.splice(2, 0, "点 ▶ 会在右下角的播放器里从那一帧播放全片逐帧标注（与原视频逐帧对应）；播放器可以收起。");
    }
    if (detail.automatic && detail.automatic.converted) {
      items[1] = "输入视频已按原比例缩放、补边并转为 25 fps。这里的帧号、坐标和标注对应转换后的视频，声音来自上传原片。";
    }
    return el(
      "section",
      { class: "panel" },
      el("h2", null, "先看这里"),
      el("ul", null, items.map((text) => el("li", null, text))),
      detail.run.notes_zh.length
        ? [el("h4", null, "这次运行的说明"), el("ul", null, detail.run.notes_zh.map((n) => el("li", null, n)))]
        : null
    );
  }

  function labelsPanel(labels) {
    if (!labels) return null;
    const rows = labels.rows.map((row) => {
      let match = "未检出";
      if (row.matched) match = row.highlight || "—";
      else if (row.covered_by) match = "没有单独成段（在 " + row.covered_by + " 的窗口里）";
      let tolerance = "—";
      if (row.matched && row.peak_within_tolerance !== null) {
        tolerance = row.peak_within_tolerance ? chip("在容差内", "good") : chip("超出容差", "bad");
      }
      return el(
        "tr",
        null,
        el("td", null, row.label),
        el("td", null, row.label_type_zh || "—"),
        el("td", { class: "num" }, row.label_peak === null ? "—" : "第 " + row.label_peak + " 帧"),
        el("td", null, match),
        el(
          "td",
          { class: "num" },
          row.matched && row.peak_error_frames !== null ? signed(row.peak_error_frames) + " 帧" : "—"
        ),
        el(
          "td",
          null,
          tolerance,
          row.tolerance_frames !== null ? el("div", { class: "reason" }, "容差 ±" + row.tolerance_frames + " 帧") : null
        ),
        el("td", null, row.matched && row.type_agrees !== null ? (row.type_agrees ? "一致" : "不一致") : "—"),
        el("td", { class: "num" }, row.window_iou === null ? "—" : row.window_iou.toFixed(2))
      );
    });
    const headers = ["标注", "类型", "标注关键帧", "对应高光", "关键帧差", "容差", "类型一致", "窗口重合（IoU）"];
    return el(
      "section",
      { class: "panel" },
      el("h2", null, "与标注对比"),
      // Where the labels come from is the label file's to say (label_eval.json source_zh), as in the report.
      el(
        "p",
        { class: "muted small" },
        (labels.source_zh || "这次运行没有记下标注的来源。") +
          "检测器的阈值只在一段开发片段上调过；下面只是检出结果和这份标注的对照，不是验证。"
      ),
      labels.sentences_zh.length ? el("ul", null, labels.sentences_zh.map((s) => el("li", null, s))) : null,
      rows.length
        ? el(
            "div",
            { class: "tablewrap" },
            el(
              "table",
              { class: "table" },
              el("thead", null, el("tr", null, headers.map((h) => el("th", { scope: "col" }, h)))),
              el("tbody", null, rows)
            )
          )
        : null
    );
  }

  function momentsTable(run, full, card) {
    if (!card.moments.length) return null;
    const rows = card.moments.map((m) =>
      el(
        "tr",
        null,
        el("td", { class: "num" }, m.clock || "—"),
        el("td", { class: "num" }, String(m.frame)),
        el("td", null, m.kind_zh),
        el("td", null, m.text_zh),
        el("td", null, playButton(run, full, m.frame, m.clock, m.seek_s, false))
      )
    );
    const headers = ["时间", "帧", "事件", "说明", "全片逐帧标注"];
    return [
      el("h4", null, "关键时刻（逐帧，原始画面上的 " + card.moments_source_zh + "）"),
      el(
        "div",
        { class: "tablewrap" },
        el(
          "table",
          { class: "table" },
          el("thead", null, el("tr", null, headers.map((h) => el("th", { scope: "col" }, h)))),
          el("tbody", null, rows)
        )
      ),
    ];
  }

  function measurementsTable(run, full, card) {
    const nodes = [el("h4", null, "测量记录（" + card.measurements.length + " 条）")];
    if (!card.measurements.length) {
      nodes.push(el("p", { class: "muted" }, "这个高光没有测量记录。"));
    } else {
      const rows = card.measurements.map((m) => {
        const missing = m.value_text === "—";
        const unit = missing || !m.unit ? "" : (m.unit === "°" ? "" : " ") + m.unit;
        const where = [];
        if (m.frame !== null) {
          where.push(
            el(
              "span",
              { class: "num" },
              "第 " + m.frame + " 帧 ",
              playButton(run, full, m.frame, null, m.seek_s, true)
            )
          );
        }
        if (m.track_id !== null) where.push(el("div", { class: "muted small" }, "轨迹 " + m.track_id));
        return el(
          "tr",
          null,
          el(
            "td",
            { title: m.evidence_id ? "证据编号 " + m.evidence_id : null },
            m.label,
            m.sign_note ? el("div", { class: "reason" }, m.sign_note) : null
          ),
          el("td", { class: "value" + (missing ? " missing" : "") }, m.value_text + unit),
          el("td", null, chip(m.tier_zh, "tier")),
          el(
            "td",
            null,
            el("span", { class: "chips" }, m.badges.map((b) => chip(b.text, b.kind))),
            m.reason ? el("div", { class: "reason" }, m.reason) : null
          ),
          el("td", null, where.length ? where : "—")
        );
      });
      const headers = ["指标", "数值", "档位", "状态", "位置"];
      nodes.push(
        el(
          "div",
          { class: "tablewrap" },
          el(
            "table",
            { class: "table" },
            el("thead", null, el("tr", null, headers.map((h) => el("th", { scope: "col" }, h)))),
            el("tbody", null, rows)
          )
        )
      );
    }
    if (card.blocked_count) {
      nodes.push(el("p", { class: "muted small" }, "另有 " + card.blocked_count + " 条 C 档记录按规定不显示。"));
    }
    return nodes;
  }

  function highlightCard(run, full, card) {
    const when = el(
      "div",
      { class: "when" },
      el(
        "span",
        { class: "num" },
        (card.start_clock || "—") + "–" + (card.end_clock || "—") + "（第 " + card.start_frame + "–" + card.end_frame + " 帧）"
      ),
      el("span", null, "·"),
      // Only the frame and its clock stay on one line: the rule after them wraps on a phone.
      el(
        "span",
        null,
        el("span", { class: "num" }, "关键帧 第 " + card.peak_frame + " 帧（" + (card.peak_clock || "—") + "）"),
        card.key_frame_rule_zh ? "，" + card.key_frame_rule_zh : null
      ),
      playButton(run, full, card.peak_frame, card.peak_clock, card.peak_seek_s, false)
    );
    const figures = [];
    if (card.keyframe) {
      figures.push(
        el(
          "figure",
          null,
          el(
            "a",
            { href: fileUrl(run, card.keyframe), target: "_blank", rel: "noopener noreferrer" },
            el("img", {
              src: fileUrl(run, card.keyframe),
              alt: card.name + " 的关键帧：原始画面上的标注",
              loading: "lazy",
            })
          ),
          el(
            "figcaption",
            null,
            "关键帧 " + card.peak_frame + "（" + (card.peak_clock || "—") + "）· 原始画面上的标注",
            card.card_image
              ? [" · ", el("a", { href: fileUrl(run, card.card_image), target: "_blank", rel: "noopener noreferrer" }, "定格数据卡")]
              : null
          )
        )
      );
    }
    if (card.clip) {
      figures.push(
        el(
          "figure",
          null,
          el("video", { src: fileUrl(run, card.clip), controls: true, preload: "metadata" }),
          el("figcaption", null, "片段：原速 + 慢放 + 定格（含原声）")
        )
      );
    }
    return el(
      "article",
      { class: "card", id: "hl-" + card.id },
      el(
        "div",
        { class: "card-head" },
        el("span", { class: "hname" }, card.name),
        el(
          "div",
          null,
          el("h3", null, card.title_zh),
          el(
            "span",
            { class: "chips" },
            chip(card.type_zh, ""),
            card.tier_zh ? chip(card.tier_zh, card.tier === "main" ? "main" : "") : null,
            card.score_text ? el("span", { class: "chip", title: "平滑后的峰值分数，只用于挑选片段" }, "评分 " + card.score_text) : null
          )
        )
      ),
      when,
      figures.length ? el("div", { class: "media" }, figures) : el("p", { class: "muted" }, "这个高光没有渲染出的图片或片段。"),
      card.summary_zh ? el("p", { class: "summary" }, card.summary_zh) : null,
      momentsTable(run, full, card),
      card.touches_zh.map((text) => el("p", { class: "small muted" }, text)),
      measurementsTable(run, full, card)
    );
  }

  async function viewRun(name, generation) {
    const detail = await api("/api/runs/" + encodeURIComponent(name));
    if (!current(generation)) return;
    const run = detail.name;
    const full = detail.full_clip;
    const parts = [el("p", { class: "crumbs" }, el("a", { href: link({}) }, "← 分析结果"))];
    parts.push(
      el(
        "div",
        { class: "page-head" },
        el("h1", null, run),
        runStatusChip(detail.run.status),
        el("span", { class: "chips" }, (detail.restrictions || []).map((tag) => chip(tag, "tag")))
      )
    );
    if (detail.banner_zh) {
      parts.push(
        el(
          "div",
          { class: "banner", role: "note" },
          el("strong", null, detail.banner_zh),
          el("span", null, "这次运行带有上面的限制标签：结果只在这台电脑上自己看，不要分享、上传或发布。")
        )
      );
    }
    if (detail.run.status === "failed") {
      parts.push(
        el(
          "div",
          { class: "alert alert-bad" },
          el("p", null, "这次分析失败了，下面只有已经写出的部分。"),
          detail.run.error.length ? el("ul", null, detail.run.error.map((e) => el("li", null, e))) : null
        )
      );
    }
    if (detail.problems.length) {
      parts.push(
        el(
          "div",
          { class: "alert alert-warn" },
          el("p", null, "有些文件没能读取："),
          el("ul", null, detail.problems.map((p) => el("li", null, p)))
        )
      );
    }
    parts.push(factsGrid(detail));
    parts.push(
      el(
        "div",
        { class: "actions" },
        detail.report
          ? el("a", { class: "btn btn-primary", href: fileUrl(run, detail.report), target: "_blank", rel: "noopener noreferrer" }, "打开完整报告")
          : null,
        full
          ? el(
              "button",
              {
                type: "button",
                class: "btn",
                onclick: () => dockPlay(run, full.path, 0, "全片逐帧标注（原始画面，与原视频逐帧对应）"),
              },
              "▶ 播放全片逐帧标注"
            )
          : el("span", { class: "muted small" }, "这次运行没有全片逐帧标注（clips/full.mp4）。")
      )
    );
    parts.push(honestyNotes(detail));
    if (detail.timeline) {
      const picture = el("picture", null);
      if (detail.timeline.dark) {
        picture.append(el("source", { media: "(prefers-color-scheme: dark)", srcset: fileUrl(run, detail.timeline.dark) }));
      }
      picture.append(
        el("img", { src: fileUrl(run, detail.timeline.light), alt: "逐帧高光分数与高光窗口的时间轴", class: "timeline" })
      );
      parts.push(
        el(
          "section",
          { class: "panel" },
          el("h2", null, "时间轴"),
          el("div", { class: "media-single" }, picture),
          el("p", { class: "muted small" }, "横轴是时间，纵轴是逐帧高光分数；橙色带是高光窗口，竖线是硬切镜。")
        )
      );
    }
    parts.push(labelsPanel(detail.labels));
    parts.push(el("h2", { class: "section-title" }, "高光片段（按时间先后编号）"));
    if (!detail.highlights.length) parts.push(el("p", { class: "muted" }, "这次运行没有高光。"));
    for (const card of detail.highlights) parts.push(highlightCard(run, full, card));
    parts.push(
      el(
        "details",
        { class: "panel" },
        el("summary", null, "这次运行的文件（" + detail.files.length + " 个，只列出能在浏览器里打开的类型）"),
        el(
          "ul",
          { class: "files" },
          detail.files.map((file) =>
            el(
              "li",
              null,
              el("a", { href: fileUrl(run, file.path), target: "_blank", rel: "noopener noreferrer" }, file.path),
              " ",
              el("span", { class: "muted" }, sizeText(file.bytes))
            )
          )
        )
      )
    );
    setView(...parts);
  }

  // ------------------------------------------------------------------ new analysis

  const FIELDS = [
    ["video", "视频文件", true, "恒定帧率的原视频；帧号以它为准。写本机的完整路径。"],
    ["audio_source", "原声文件", false, "带原始声音的文件；不填就用视频本身的声音。"],
    ["sngs_dir", "插件导出文件夹", true, "plugin-sngs 的导出（detections.jsonl、calib.jsonl、plugin-receipt.json）。"],
    ["ball", "球检测文件", true, "球检测插件写出的 ball_raw.jsonl。"],
    ["labels", "标注文件", false, "填了会在结果里和高光对比；只是对照，不是验证。"],
  ];

  function field(name, label, required, hint, value) {
    const id = "f-" + name;
    return el(
      "div",
      { class: "field" },
      el("label", { for: id }, label, required ? null : el("span", { class: "opt" }, "可选")),
      el("input", {
        id,
        name,
        type: "text",
        value: value || "",
        required: required || null,
        spellcheck: "false",
        autocomplete: "off",
      }),
      el("div", { class: "hint" }, hint)
    );
  }

  function suggestedName() {
    const now = new Date();
    const pad = (n) => String(n).padStart(2, "0");
    return (
      "run-" + now.getFullYear() + pad(now.getMonth() + 1) + pad(now.getDate()) + "-" + pad(now.getHours()) + pad(now.getMinutes())
    );
  }

  const JOB_STATUS = {
    running: ["运行中", "run"],
    succeeded: ["完成", "good"],
    refused: ["被拒绝", "bad"],
    failed: ["失败", "bad"],
    interrupted: ["已中断", "warn"],
    unknown: ["结果不明", ""],
  };

  function jobStatusChip(status) {
    const [text, kind] = JOB_STATUS[status] || [status, ""];
    return chip(text, kind);
  }

  function uploadVideo(file, onProgress, onRequest) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "/api/uploads");
      xhr.setRequestHeader(TOKEN_HEADER, state.token);
      xhr.setRequestHeader("Content-Type", "application/octet-stream");
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(event.loaded / event.total);
      };
      xhr.onload = () => {
        let data;
        try { data = JSON.parse(xhr.responseText); } catch (_) { data = {}; }
        if (xhr.status >= 200 && xhr.status < 300) resolve(data.video);
        else reject(failure(data.error_zh || "视频传入失败，请重试。", data.problems_zh));
      };
      xhr.onerror = () => reject(failure("连接中断，视频没有传入完成。请检查本机服务。"));
      xhr.onabort = () => reject(failure("已取消传入，可以重新选择视频。"));
      onRequest(xhr);
      xhr.send(file);
    });
  }

  async function viewNew(generation) {
    const data = await api("/api/jobs");
    if (!current(generation)) return;
    const automatic = state.session.automatic || { available: false, message_zh: "自动识别未配置。" };
    const last = data.jobs.length ? data.jobs[0].request : null;
    const value = (name) => (last && last[name]) || "";
    const teams = (last && last.team_names) || {};
    const status = el("div", { "aria-live": "polite" });
    const submit = el("button", { type: "submit", class: "btn btn-primary" }, "开始自动分析");
    const manual = el("input", { id: "f-manual", type: "checkbox" });
    const picker = el("input", { id: "video-picker", type: "file", accept: "video/*,.mkv,.m4v,.avi" });
    const selection = el("p", { class: "muted small", "aria-live": "polite" }, "尚未选择视频");
    const progress = el("progress", { max: "1", value: "0", hidden: true, "aria-label": "视频传入进度" });
    let request = null;
    let savedUpload = null;
    const cancelUpload = el("button", { type: "button", class: "btn", hidden: true,
      onclick: () => { if (request) request.abort(); } }, "取消传入");
    picker.addEventListener("change", () => {
      savedUpload = null;
      const file = picker.files[0];
      selection.textContent = file ? file.name + " · " + (file.size / 1024 / 1024).toFixed(1) + " MB" : "尚未选择视频";
    });
    const imported = el("div", { class: "form manual-fields", hidden: true },
      FIELDS.filter(([name]) => ["sngs_dir", "ball", "labels"].includes(name))
        .map(([name, label, required, hint]) => field(name, label, required, hint, value(name))));
    const syncMode = () => {
      imported.hidden = !manual.checked;
      picker.disabled = manual.checked;
      submit.textContent = manual.checked ? "分析已有预测" : "开始自动分析";
      submit.disabled = Boolean(data.current) || (!manual.checked && !automatic.available);
    };
    manual.addEventListener("change", syncMode);
    const form = el("form", { class: "form", novalidate: true },
      el("div", { class: "video-select field" },
        el("label", { for: "video-picker" }, "选择足球视频"), picker, selection,
        el("div", { class: "hint" }, "文件仅传到这台电脑。支持 60 秒以内、512 MB 以内的短片。")),
      el("details", null, el("summary", null, "或填写本地视频完整路径"),
        field("video", "视频路径", false, "已选择文件时优先使用所选文件；手动导入模式请填写路径。", "")),
      el("p", { class: "workflow-note" }, "准备视频 → 识别球员与足球 → 筛选高光 → 分析动作 → 生成报告"),
      el("div", { class: "alert " + (automatic.available ? "alert-warn" : "alert-bad") },
        automatic.message_zh, el("p", { class: "small" }, "处理时请保持本机服务运行；可关闭页面后用原地址回来查看。运行成功不代表测量准确，3D 只作参考。")),
      el("details", { class: "advanced" }, el("summary", null, "高级选项"),
        el("div", { class: "form" },
          el("label", { class: "check-label", for: "f-manual" }, manual, "导入已生成的追踪、标定和球检测数据"), imported,
          field("audio_source", "原声文件", false, "仅手动导入时使用；自动模式从所选原视频提取声音。", ""),
          el("div", { class: "field" }, el("label", null, "球队名（可选）"),
            el("div", { class: "pair" },
              el("input", { id: "f-team-left", type: "text", placeholder: "左侧球队", value: teams.left || "", "aria-label": "左侧球队" }),
              el("input", { id: "f-team-right", type: "text", placeholder: "右侧球队", value: teams.right || "", "aria-label": "右侧球队" }))),
          field("allow_licenses", "额外接受的许可证", false, "自动模式使用本机已配置的模型许可。手动导入可填写 SPDX 标识，多个用逗号分隔。", ""),
          field("out_name", "输出名", true, "自动生成；可改用英文字母、数字和 . _ -，已有结果不会覆盖。", suggestedName() + "-" + String(new Date().getSeconds()).padStart(2, "0")))),
      el("div", { class: "actions" }, submit, cancelUpload), progress, status);
    syncMode();
    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const read = (id) => $(id).value.trim();
      const licenses = read("f-allow_licenses").split(/[,，、;；]+/).map((x) => x.trim()).filter(Boolean);
      const isManual = manual.checked;
      const file = picker.files[0];
      submit.disabled = true;
      manual.disabled = true;
      picker.disabled = true;
      try {
        let video = read("f-video");
        if (!isManual && file) {
          if (!file.size || file.size > (automatic.max_upload_bytes || 512 * 1024 * 1024)) {
            throw failure("文件为空或超过 512 MB，请先截取短片。");
          }
          if (!savedUpload) {
            progress.hidden = false;
            cancelUpload.hidden = false;
            fill(status, el("p", null, "正在将视频传入本机……"));
            savedUpload = await uploadVideo(file, (ratio) => {
              progress.value = ratio;
              fill(status, el("p", null, "视频传入本机 " + Math.round(ratio * 100) + "%"));
            }, (xhr) => { request = xhr; });
          }
          video = savedUpload;
        }
        if (!video) throw failure(isManual ? "请填写视频的完整路径。" : "请选择视频，或填写本地视频完整路径。");
        const payload = {
          mode: isManual ? "manual" : "auto", video,
          team_names: read("f-team-left") || read("f-team-right")
            ? { left: read("f-team-left"), right: read("f-team-right") } : null,
          allow_licenses: licenses, out_name: read("f-out_name"),
        };
        if (isManual) Object.assign(payload, {
          audio_source: read("f-audio_source") || null, sngs_dir: read("f-sngs_dir"),
          ball: read("f-ball"), labels: read("f-labels") || null,
        });
        fill(status, el("p", null, "正在检查视频并启动分析……"));
        const job = await api("/api/jobs", { method: "POST", json: payload });
        location.hash = link({ job: job.id });
      } catch (error) {
        fill(status, errorBox(error));
      } finally {
        request = null;
        cancelUpload.hidden = true;
        progress.hidden = true;
        manual.disabled = false;
        syncMode();
      }
    });
    const parts = [el("div", { class: "page-head" }, el("h1", null, "新建视频分析"))];
    if (data.current) parts.push(el("div", { class: "alert alert-warn" }, "已有分析正在运行：",
      el("a", { href: link({ job: data.current }) }, "查看进度或停止")));
    parts.push(el("section", { class: "panel" }, form));
    if (data.jobs.length) {
      parts.push(
        el(
          "section",
          { class: "panel" },
          el("h2", null, "最近的分析任务"),
          el(
            "div",
            { class: "tablewrap" },
            el(
              "table",
              { class: "table" },
              el("thead", null, el("tr", null, ["开始时间", "输出名", "状态", ""].map((h) => el("th", { scope: "col" }, h)))),
              el(
                "tbody",
                null,
                data.jobs.map((job) =>
                  el(
                    "tr",
                    null,
                    el("td", { class: "num" }, fmtTime(job.created_at)),
                    el("td", null, job.out_name),
                    el("td", null, jobStatusChip(job.status)),
                    el("td", null, el("a", { href: link({ job: job.id }) }, "详情"))
                  )
                )
              )
            )
          )
        )
      );
    }
    setView(...parts);
  }

  // ------------------------------------------------------------------ one job

  const STAGE_STATE = {
    done: ["✓", "完成"],
    skipped: ["✓", "沿用上次结果"],
    kept: ["✓", "保留"],
    running: ["…", "进行中"],
    failed: ["✕", "失败"],
    pending: ["", "等待"],
  };

  function jobView(id) {
    const title = el("div", { class: "page-head" }, el("h1", null, "分析任务"));
    const message = el("div");
    const stages = el("ol", { class: "stages" });
    const actions = el("div", { class: "actions" });
    const log = el("pre", { class: "log" });
    const logBox = el("details", { open: true }, el("summary", null, "日志的最后几行"), log);
    const request = el("details", null, el("summary", null, "这次分析的输入"));
    setView(
      el("p", { class: "crumbs" }, el("a", { href: link({ view: "new" }) }, "← 新建分析")),
      title,
      el("section", { class: "panel" }, message, stages, actions),
      el("section", { class: "panel" }, logBox, request)
    );
    return (job) => {
      fill(title, el("h1", null, "分析任务 " + job.out_name), jobStatusChip(job.status));
      const problems = job.problems_zh || [];
      const quiet = job.status === "running" || job.status === "succeeded";
      fill(
        message,
        el("p", null, job.message_zh),
        problems.length ? el("ul", null, problems.map((p) => el("li", null, p))) : null,
        !problems.length && !quiet ? el("p", { class: "muted small" }, "详细原因见下面的日志。") : null,
        el(
          "p",
          { class: "muted small" },
          "开始：" + fmtTime(job.created_at) +
            (job.finished_at ? " · 结束：" + fmtTime(job.finished_at) : "") +
            (job.exit_code !== null ? " · 退出码 " + job.exit_code : "")
        )
      );
      fill(
        stages,
        job.stages.map((stage) => {
          let [mark, text] = STAGE_STATE[stage.state] || ["", stage.state];
          if (stage.state === "pending" && job.status !== "running") text = "未运行";
          return el(
            "li",
            { class: "stage stage-" + stage.state },
            el("span", { class: "mark", "aria-hidden": "true" }, mark),
            el("span", null, stage.name_zh),
            el("span", { class: "state" }, text)
          );
        })
      );
      fill(
        actions,
        job.status === "running" ? el("button", {
          type: "button", class: "btn", onclick: async (event) => {
            const button = event.currentTarget;
            button.disabled = true;
            button.textContent = "正在停止……";
            try {
              await api("/api/jobs/" + encodeURIComponent(job.id) + "/cancel", { method: "POST", json: {} });
              await route();
            } catch (error) {
              fill(message, errorBox(error));
              button.disabled = false;
              button.textContent = "停止分析";
            }
          }
        }, "停止分析") : null,
        job.run_ready
          ? el(
              "a",
              { class: "btn btn-primary", href: link({ run: job.out_name }) },
              job.status === "succeeded" ? "查看结果" : "查看这次运行写出的部分"
            )
          : null
      );
      const atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 8;
      log.textContent = job.log_tail.join("\n") || "（还没有输出）";
      if (atBottom) log.scrollTop = log.scrollHeight;
      logBox.querySelector("summary").textContent =
        "日志的最后几行（完整日志在运行目录下的 " + job.log_file + "）";
      const req = job.request || {};
      fill(
        request,
        el("summary", null, "这次分析的输入"),
        el(
          "ul",
          { class: "small" },
          Object.entries(state.session.fields_zh).map(([key, label]) => {
            let shown = req[key];
            if (Array.isArray(shown)) shown = shown.join(", ");
            else if (shown && typeof shown === "object") shown = Object.entries(shown).map(([k, v]) => k + "=" + v).join(", ");
            return el("li", null, label + "：", el("code", null, shown || "—"));
          })
        )
      );
    };
  }

  async function viewJob(id, generation) {
    const update = jobView(id);
    const tick = async () => {
      let job;
      try {
        job = await api("/api/jobs/" + encodeURIComponent(id));
      } catch (error) {
        if (current(generation)) setView(errorBox(error));
        return;
      }
      if (!current(generation)) return;
      update(job);
      if (job.status === "running") state.timer = window.setTimeout(tick, POLL_MS);
    };
    await tick();
  }

  // ------------------------------------------------------------------ routing

  async function route() {
    window.clearTimeout(state.timer);
    const generation = ++state.generation;
    const params = hashParams();
    const token = params.get("t");
    try {
      if (token && token !== state.token) {
        state.token = token;
        state.session = await api("/api/session");
        renderPrivacy();
      }
      if (params.get("run")) {
        renderNav("");
        await viewRun(params.get("run"), generation);
      } else if (params.get("job")) {
        renderNav("new");
        await viewJob(params.get("job"), generation);
      } else if (params.get("view") === "new") {
        renderNav("new");
        await viewNew(generation);
      } else {
        renderNav("runs");
        await viewRuns(generation);
      }
    } catch (error) {
      if (current(generation)) setView(errorBox(error));
    }
  }

  async function start() {
    state.token = hashParams().get("t") || "";
    $("dock-close").addEventListener("click", closeDock);
    $("dock-fold").addEventListener("click", () => foldDock(!$("dock").classList.contains("folded")));
    if ("ResizeObserver" in window) new ResizeObserver(dockSpace).observe($("dock"));
    window.addEventListener("resize", dockSpace);
    if (!state.token) {
      setView(
        el(
          "div",
          { class: "alert alert-bad", role: "alert" },
          el("p", null, "缺少访问令牌。"),
          el("p", null, "请用启动服务时终端里打印的完整地址（带 #t=…）打开本页。")
        )
      );
      return;
    }
    try {
      state.session = await api("/api/session");
    } catch (error) {
      setView(errorBox(error));
      return;
    }
    renderPrivacy();
    window.addEventListener("hashchange", () => {
      closeDock();
      route();
    });
    await route();
  }

  start();
})();
