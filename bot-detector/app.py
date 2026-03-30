#!/usr/bin/env python3
"""Bot-or-Not. Run: python3 app.py"""
import sys, threading, os
from pathlib import Path
from tkinter import filedialog, ttk
import tkinter as tk

sys.path.insert(0, str(Path(__file__).parent))
import customtkinter as ctk
from src.detector import score_dataset
from src.utils import load_dataset, competition_score

ctk.set_appearance_mode("light")
ARTIFACTS_DIR = str(Path(__file__).parent / "artifacts")

W      = "#ffffff"
BG     = "#f5f5f7"
SUB    = "#86868b"
TEXT   = "#1d1d1f"
BLUE   = "#0071e3"
RED    = "#ff3b30"
GRN    = "#34c759"
ORANGE = "#ff9500"
PURPLE = "#af52de"
LINE   = "#d2d2d7"
ROW    = "#fafafa"

def F(size, bold=False):
    return ctk.CTkFont(family="Inter", size=size, weight="bold" if bold else "normal")

SIGNALS = [
    ("cross_exact_dup_rate",    94, "Posts identical text as another account"),
    ("cross_near_dup_rate",     91, "Posts near-identical text as another account"),
    ("coordination_score",      88, "Coordinated activity with other accounts"),
    ("exact_dup_rate",          85, "Repeats the same post verbatim"),
    ("near_dup_rate",           82, "Repeats near-identical posts"),
    ("default_profile_image",   79, "Never set a profile picture"),
    ("default_profile",         76, "Never customised the profile"),
    ("hashtag_burst_score",     73, "Hashtags appear in sudden coordinated bursts"),
    ("inter_post_std",          70, "Clock-like, robotic gap between posts"),
    ("posts_per_day",           68, "Posts far more than a normal person would"),
    ("shared_hashtag_score",    65, "Uses the exact same hashtags as other accounts"),
    ("z_score",                 62, "Statistical outlier in tweet volume"),
    ("ff_ratio_log",            60, "Follower / following ratio looks unnatural"),
    ("account_age_days",        57, "Very new account"),
    ("hashtag_rate",            54, "Almost every post has hashtags"),
    ("url_rate",                51, "Almost every post contains a link"),
    ("compression_ratio",       48, "Text is highly repetitive"),
    ("type_token_ratio",        45, "Very limited vocabulary"),
    ("hour_entropy",            43, "Posts evenly across all 24 hours"),
    ("followers_count",         41, "Suspiciously low or inflated follower count"),
    ("statuses_count",          39, "Lifetime tweet count looks automated"),
    ("night_post_rate",         37, "Posts heavily during sleeping hours"),
    ("source_entropy",          35, "Switches between posting clients unusually"),
    ("retweet_rate",            33, "Mostly retweets, rarely original content"),
    ("caps_rate",               30, "Excessive use of uppercase text"),
    ("avg_post_length",         28, "All posts are unusually similar in length"),
    ("burst_ratio",             26, "Activity comes in dense automated bursts"),
    ("mention_rate",            24, "Mentions other accounts very frequently"),
    ("bot_signal_count",        22, "Total individual bot signals triggered"),
    ("content_diversity_score", 20, "All content is about one narrow topic"),
]


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Bot or Not")
        self.geometry("1260x820")
        self.minsize(960, 620)
        self.configure(fg_color=BG)
        self._flagged_ids  = []
        self._bot_records  = []
        self._hum_records  = []
        self._all_records  = []   # flat list of all records after detection
        self._style_ttk()
        self._build()

    # ── ttk styles ───────────────────────────────────────────────────────
    def _style_ttk(self):
        s = ttk.Style(self)
        s.theme_use("default")
        for name in ("Bot.Treeview", "Hum.Treeview"):
            s.configure(name,
                background=W, foreground=TEXT,
                fieldbackground=W, rowheight=48,
                font=("Inter", 12), borderwidth=0,
            )
            s.configure(f"{name}.Heading",
                background=W, foreground=SUB,
                font=("Inter", 10), relief="flat", borderwidth=0,
            )
            s.map(name,
                background=[("selected", BG)],
                foreground=[("selected", TEXT)],
            )
        s.configure("Thin.Vertical.TScrollbar",
            background=W, troughcolor=W, borderwidth=0,
            arrowcolor=LINE, width=4,
        )

    # ── Layout ───────────────────────────────────────────────────────────
    def _build(self):
        # Top bar
        bar = ctk.CTkFrame(self, fg_color=W, corner_radius=0, height=62)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        ctk.CTkLabel(bar, text="Bot or Not", font=F(17, bold=True),
                     text_color=TEXT).pack(side="left", padx=28)

        self._dl = ctk.CTkButton(
            bar, text="Export Bots",
            font=F(13), fg_color=BLUE, hover_color="#005bbf",
            text_color=W, corner_radius=20,
            width=124, height=36,
            state="disabled", command=self._download,
        )
        self._dl.pack(side="right", padx=(4, 24))

        self._cmp_btn = ctk.CTkButton(
            bar, text="Compare Ground Truth",
            font=F(13), fg_color=PURPLE, hover_color="#8e3fbd",
            text_color=W, corner_radius=20,
            width=178, height=36,
            state="disabled", command=self._load_ground_truth,
        )
        self._cmp_btn.pack(side="right", padx=4)

        self._back_btn = ctk.CTkButton(
            bar, text="← New Dataset",
            font=F(13), fg_color=BG, hover_color=LINE,
            text_color=SUB, corner_radius=20,
            width=124, height=36,
            state="disabled", command=self._reset,
        )
        self._back_btn.pack(side="right", padx=4)

        self._status = ctk.CTkLabel(bar, text="", font=F(12), text_color=SUB)
        self._status.pack(side="right", padx=8)

        ctk.CTkFrame(self, fg_color=LINE, height=1, corner_radius=0).pack(fill="x")

        # Body
        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=20)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=0, minsize=256)
        body.rowconfigure(2, weight=1)

        # Stat pills (row 0)
        stats = ctk.CTkFrame(body, fg_color="transparent")
        stats.grid(row=0, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._s_bots   = self._pill(stats, "Bots",   "—", RED)
        self._s_humans = self._pill(stats, "Humans", "—", GRN)
        self._s_total  = self._pill(stats, "Total",  "—", TEXT)

        # Verification stats bar (row 1) — hidden until ground truth loaded
        self._verify_frame = ctk.CTkFrame(body, fg_color="transparent")
        self._verify_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 10))
        self._verify_frame.grid_remove()  # hidden initially

        vf_inner = ctk.CTkFrame(self._verify_frame, fg_color=W, corner_radius=14)
        vf_inner.pack(side="left", ipadx=18, ipady=10)

        ctk.CTkLabel(vf_inner, text="VERIFICATION", font=F(9), text_color=PURPLE).pack(
            side="left", padx=(12, 14))

        self._v_acc   = self._verify_pill(vf_inner, "Score %",               "—", GRN)
        self._v_score = self._verify_pill(vf_inner, "Score",                "—", TEXT)
        self._v_tp    = self._verify_pill(vf_inner, "Bots Caught  +2 each", "—", GRN)
        self._v_fp    = self._verify_pill(vf_inner, "Humans Wrongly Flagged  −6 each", "—", ORANGE)
        self._v_fn    = self._verify_pill(vf_inner, "Bots Missed  −2 each", "—", PURPLE)

        self._reset_btn = ctk.CTkButton(
            self._verify_frame, text="Clear",
            font=F(11), fg_color=LINE, hover_color="#c0c0c5",
            text_color=SUB, corner_radius=14,
            width=60, height=32,
            command=self._clear_comparison,
        )
        self._reset_btn.pack(side="left", padx=(10, 0))

        # Lists (row 2)
        self._bot_col = self._list_col(body, "Bots",   RED, 0, "Bot.Treeview")
        self._hum_col = self._list_col(body, "Humans", GRN, 1, "Hum.Treeview")

        # Signal sidebar (row 2, col 2)
        self._build_signals(body, 2)

        # Welcome overlay
        self._welcome = ctk.CTkFrame(body, fg_color=BG, corner_radius=0)
        self._welcome.grid(row=2, column=0, columnspan=2, sticky="nsew")
        inner = ctk.CTkFrame(self._welcome, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")
        ctk.CTkLabel(inner, text="Upload a dataset to begin.",
                     font=F(16), text_color=SUB).pack(pady=(0, 22))
        ctk.CTkButton(inner, text="Choose File",
                      font=F(14, bold=True),
                      fg_color=BLUE, hover_color="#005bbf",
                      text_color=W, corner_radius=22,
                      width=160, height=46,
                      command=self._load).pack()

    # ── Small inline pill for verify bar ────────────────────────────────
    def _verify_pill(self, parent, label, value, color):
        wrap = ctk.CTkFrame(parent, fg_color="transparent")
        wrap.pack(side="left", padx=(0, 16))
        v = ctk.CTkLabel(wrap, text=value, font=F(20, bold=True), text_color=color)
        v.pack()
        ctk.CTkLabel(wrap, text=label, font=F(9), text_color=SUB).pack()
        return v

    # ── Pill ─────────────────────────────────────────────────────────────
    def _pill(self, parent, label, value, color):
        card = ctk.CTkFrame(parent, fg_color=W, corner_radius=14)
        card.pack(side="left", padx=(0, 10), ipadx=18, ipady=12)
        v = ctk.CTkLabel(card, text=value, font=F(30, bold=True), text_color=color)
        v.pack()
        ctk.CTkLabel(card, text=label.upper(), font=F(9), text_color=SUB).pack()
        return v

    # ── Fast native list ─────────────────────────────────────────────────
    def _list_col(self, parent, title, accent, col, style):
        outer = ctk.CTkFrame(parent, fg_color=W, corner_radius=16)
        outer.grid(row=2, column=col, sticky="nsew", padx=(0, 14) if col == 0 else (0, 14))

        # Header
        hdr = ctk.CTkFrame(outer, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(18, 10))
        ctk.CTkLabel(hdr, text=title, font=F(14, bold=True), text_color=TEXT).pack(side="left")
        cnt = ctk.CTkLabel(hdr, text="", font=F(13), text_color=SUB)
        cnt.pack(side="right")

        # Search
        sv = tk.StringVar()
        e = ctk.CTkEntry(outer, textvariable=sv,
                         placeholder_text="Search…", font=F(12),
                         fg_color=BG, border_color=LINE,
                         text_color=TEXT, corner_radius=10, height=38)
        e.pack(fill="x", padx=14, pady=(0, 12))

        # Treeview
        wrap = tk.Frame(outer, bg=W, bd=0, highlightthickness=0)
        wrap.pack(fill="both", expand=True, padx=6, pady=(0, 12))

        tree = ttk.Treeview(wrap, style=style,
                            columns=("user", "score"),
                            show="headings", selectmode="browse")
        tree.heading("user",  text="Username")
        tree.heading("score", text="Score")
        tree.column("user",  anchor="w", stretch=True)
        tree.column("score", anchor="e", width=80, stretch=False)

        # Normal tags
        tree.tag_configure("bot",    foreground=RED)
        tree.tag_configure("human",  foreground=GRN)
        # Verification tags
        tree.tag_configure("tp",     foreground=GRN)
        tree.tag_configure("fp",     foreground=ORANGE)
        tree.tag_configure("fn",     foreground=PURPLE)

        vsb = ttk.Scrollbar(wrap, style="Thin.Vertical.TScrollbar",
                            orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        tree.pack(fill="both", expand=True)

        outer._cnt      = cnt
        outer._tree     = tree
        outer._records  = []
        outer._tag      = "bot" if accent == RED else "human"
        outer._tag_map  = {}   # user_id -> tag override (for comparison mode)
        sv.trace_add("write", lambda *_: self._filter(outer, sv))
        outer._sv = sv
        return outer

    def _fill(self, col, records):
        col._records  = records
        col._tag_map  = {}
        col._cnt.configure(text=str(len(records)))
        t = col._tree
        t.delete(*t.get_children())
        tag = col._tag
        for r in records:
            name = r.get("username") or str(r.get("user_id", ""))
            t.insert("", "end", iid=r["user_id"],
                     values=(name, f"{int(r['bot_score']*100)}%"), tags=(tag,))

    def _fill_tagged(self, col, records, tag_map):
        """Fill treeview using per-record tag overrides from comparison."""
        col._records  = records
        col._tag_map  = tag_map
        col._cnt.configure(text=str(len(records)))
        t = col._tree
        t.delete(*t.get_children())
        default = col._tag
        for r in records:
            uid  = r["user_id"]
            name = r.get("username") or uid
            tag  = tag_map.get(uid, default)
            label, status = self._score_label(r, tag)
            t.insert("", "end", iid=uid,
                     values=(name, label), tags=(tag,))

    def _score_label(self, r, tag):
        pct = f"{int(r['bot_score']*100)}%"
        if tag == "tp":
            return f"✓ caught  {pct}", "caught"
        if tag == "fp":
            return f"✗ wrong flag  {pct}", "wrong"
        if tag == "fn":
            return f"! missed  {pct}", "missed"
        return pct, ""

    def _filter(self, col, sv):
        q = sv.get().lower().strip()
        t = col._tree
        t.delete(*t.get_children())
        default = col._tag
        for r in col._records:
            name = r.get("username") or str(r.get("user_id", ""))
            if not q or q in name.lower():
                tag = col._tag_map.get(r["user_id"], default)
                label, _ = self._score_label(r, tag)
                t.insert("", "end", iid=r["user_id"],
                         values=(name, label), tags=(tag,))

    # ── Signal sidebar ───────────────────────────────────────────────────
    def _build_signals(self, parent, col):
        outer = ctk.CTkFrame(parent, fg_color=W, corner_radius=16)
        outer.grid(row=2, column=col, sticky="nsew")

        ctk.CTkLabel(outer, text="Signals", font=F(14, bold=True),
                     text_color=TEXT).pack(anchor="w", padx=20, pady=(18, 8))

        wrap = tk.Frame(outer, bg=W, bd=0, highlightthickness=0)
        wrap.pack(fill="both", expand=True, padx=4, pady=(0, 12))

        txt = tk.Text(wrap, bg=W, fg=TEXT,
                      font=("Inter", 11), relief="flat",
                      wrap="word", cursor="arrow",
                      state="disabled", padx=14, pady=4,
                      spacing1=0, spacing3=2,
                      highlightthickness=0)
        vsb = ttk.Scrollbar(wrap, style="Thin.Vertical.TScrollbar",
                            orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        txt.pack(fill="both", expand=True)

        txt.tag_configure("name", font=("Inter", 11, "bold"),
                          foreground=TEXT, spacing1=10)
        txt.tag_configure("hi",   font=("Inter", 10), foreground=BLUE)
        txt.tag_configure("mid",  font=("Inter", 10), foreground=SUB)
        txt.tag_configure("lo",   font=("Inter", 10), foreground=LINE)
        txt.tag_configure("desc", font=("Inter", 10), foreground=SUB, spacing3=6)

        txt.configure(state="normal")
        for name, imp, desc in SIGNALS:
            tag = "hi" if imp >= 60 else ("mid" if imp >= 40 else "lo")
            bar = "█" * (imp // 10) + "░" * (10 - imp // 10)
            txt.insert("end", f"{name}\n", "name")
            txt.insert("end", f"{bar}  {imp}%\n", tag)
            txt.insert("end", f"{desc}\n", "desc")
        txt.configure(state="disabled")

    # ── Load dataset ─────────────────────────────────────────────────────
    def _load(self):
        path = filedialog.askopenfilename(
            title="Select dataset",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
        )
        if not path:
            return
        self._status.configure(text="Analyzing…")
        self.update_idletasks()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        try:
            dataset = load_dataset(path)
            _, _, records = score_dataset(dataset, ARTIFACTS_DIR)
            bots   = sorted([r for r in records if     r["flagged"]], key=lambda r: -r["bot_score"])
            humans = sorted([r for r in records if not r["flagged"]], key=lambda r:  r["bot_score"])
            self._flagged_ids = [r["user_id"] for r in records if r["flagged"]]
            self._all_records = records
            self.after(0, lambda: self._show(bots, humans, len(records)))
        except Exception as ex:
            self.after(0, lambda: self._status.configure(text=f"Error: {ex}"))

    def _show(self, bots, humans, total):
        if self._welcome.winfo_exists():
            self._welcome.grid_forget()
        self._s_bots.configure(text=str(len(bots)))
        self._s_humans.configure(text=str(len(humans)))
        self._s_total.configure(text=str(total))
        self._status.configure(text="")
        self._bot_records = bots
        self._hum_records = humans
        self._fill(self._bot_col, bots)
        self._fill(self._hum_col, humans)
        self._dl.configure(state="normal")
        self._cmp_btn.configure(state="normal")
        self._back_btn.configure(state="normal")
        # Reset any prior comparison
        self._verify_frame.grid_remove()

    # ── Ground truth comparison ───────────────────────────────────────────
    def _load_ground_truth(self):
        path = filedialog.askopenfilename(
            title="Select ground truth bots file",
            filetypes=[("Text file", "*.txt"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                ground_truth = {line.strip() for line in f if line.strip()}
            self._apply_comparison(ground_truth)
        except Exception as ex:
            self._status.configure(text=f"Error loading ground truth: {ex}")

    def _apply_comparison(self, ground_truth: set):
        predicted = set(self._flagged_ids)
        all_ids   = {r["user_id"] for r in self._all_records}

        tp = len(predicted & ground_truth)
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)
        score = competition_score(tp, fp, fn)

        # Build tag maps
        bot_tag_map = {}
        for r in self._bot_records:
            uid = r["user_id"]
            bot_tag_map[uid] = "tp" if uid in ground_truth else "fp"

        hum_tag_map = {}
        for r in self._hum_records:
            uid = r["user_id"]
            hum_tag_map[uid] = "fn" if uid in ground_truth else "human"

        # Update treeviews
        self._fill_tagged(self._bot_col, self._bot_records, bot_tag_map)
        self._fill_tagged(self._hum_col, self._hum_records, hum_tag_map)

        # Competition-weighted accuracy: score / max_possible × 100
        # max_possible = 2 × total actual bots (perfect recall, zero FP)
        max_possible = 2 * (tp + fn)
        acc_pct = (score / max_possible * 100) if max_possible > 0 else 100.0
        acc_color = GRN if acc_pct >= 90 else (ORANGE if acc_pct >= 50 else RED)

        # Show verification bar
        score_color = GRN if score >= 0 else RED
        self._v_acc.configure(text=f"{acc_pct:+.0f}%", text_color=acc_color)
        self._v_score.configure(text=f"{score:+d}", text_color=score_color)
        self._v_tp.configure(text=str(tp))
        self._v_fp.configure(text=str(fp))
        self._v_fn.configure(text=str(fn))
        self._verify_frame.grid()

        self._status.configure(
            text=f"Score %: {acc_pct:+.0f}%  •  {tp} bots caught  •  {fp} humans wrongly flagged  •  {fn} bots missed"
        )

    def _clear_comparison(self):
        self._fill(self._bot_col, self._bot_records)
        self._fill(self._hum_col, self._hum_records)
        self._verify_frame.grid_remove()
        self._status.configure(text="")

    # ── Reset / back ─────────────────────────────────────────────────────
    def _reset(self):
        self._flagged_ids = []
        self._bot_records = []
        self._hum_records = []
        self._all_records = []
        self._fill(self._bot_col, [])
        self._fill(self._hum_col, [])
        self._verify_frame.grid_remove()
        self._s_bots.configure(text="—")
        self._s_humans.configure(text="—")
        self._s_total.configure(text="—")
        self._status.configure(text="")
        self._dl.configure(state="disabled")
        self._cmp_btn.configure(state="disabled")
        self._back_btn.configure(state="disabled")
        # Show welcome overlay again
        self._welcome.grid(row=2, column=0, columnspan=2, sticky="nsew")

    # ── Export ───────────────────────────────────────────────────────────
    def _download(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text file", "*.txt")],
            initialfile="detections.txt",
        )
        if not path:
            return
        with open(path, "w") as f:
            for uid in self._flagged_ids:
                f.write(f"{uid}\n")
        self._status.configure(text=f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    App().mainloop()
