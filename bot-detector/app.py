#!/usr/bin/env python3
"""FlagR — bot detection desktop app. Run: python3 app.py"""
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

def F(size, bold=False):
    return ctk.CTkFont(family="Inter", size=size, weight="bold" if bold else "normal")

def TF(size, bold=False):
    return ("Inter", size, "bold") if bold else ("Inter", size)

TAG_COLOR = {"bot": RED, "human": GRN, "tp": GRN, "fp": ORANGE, "fn": PURPLE}

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


def _scroll_units(delta: int) -> int:
    """Normalize mouse-wheel delta across platforms.
    Windows sends ±120 per notch; macOS sends ±1–5.
    Always returns a non-zero integer in the correct direction."""
    if abs(delta) >= 120:
        return int(-delta / 120)          # Windows / standard wheel
    return -1 if delta > 0 else 1         # macOS trackpad / Magic Mouse


# ── Pill-shaped scrollbar ─────────────────────────────────────────────────────
class RoundScrollbar(tk.Canvas):
    THUMB       = "#c7c7cc"
    THUMB_HOVER = "#a0a0a8"

    def __init__(self, master, command, **kw):
        kw.setdefault("bg", W); kw.setdefault("width", 6)
        kw.setdefault("bd", 0); kw.setdefault("highlightthickness", 0)
        super().__init__(master, **kw)
        self._cmd = command; self._first = 0.0; self._last = 1.0
        self._drag_y = None; self._drag_first = 0.0; self._hover = False
        self.bind("<Configure>",       lambda e: self._draw())
        self.bind("<ButtonPress-1>",   self._press)
        self.bind("<B1-Motion>",       self._drag)
        self.bind("<ButtonRelease-1>", lambda e: setattr(self, "_drag_y", None))
        self.bind("<Enter>",           lambda e: self._set_hover(True))
        self.bind("<Leave>",           lambda e: self._set_hover(False))

    def set(self, first, last):
        self._first = float(first); self._last = float(last); self._draw()

    def _thumb_bounds(self):
        h = max(1, self.winfo_height()); w = self.winfo_width()
        y1 = int(h * self._first); y2 = int(h * self._last)
        if y2 - y1 < w * 2: mid = (y1+y2)//2; y1 = max(0, mid-w); y2 = min(h, mid+w)
        return y1, y2

    def _draw(self):
        self.delete("all")
        w = self.winfo_width(); h = self.winfo_height()
        if w < 2 or h < 2 or self._last - self._first >= 0.999: return
        y1, y2 = self._thumb_bounds()
        col = self.THUMB_HOVER if self._hover else self.THUMB
        # Pill shape: top oval + bottom oval + middle rect
        self.create_oval(0, y1, w, y1+w, fill=col, outline="")
        self.create_oval(0, y2-w, w, y2,  fill=col, outline="")
        if y2 - y1 > w:
            self.create_rectangle(0, y1+w//2, w, y2-w//2, fill=col, outline="")

    def _set_hover(self, v): self._hover = v; self._draw()

    def _press(self, e):
        y1, y2 = self._thumb_bounds()
        if y1 <= e.y <= y2: self._drag_y = e.y; self._drag_first = self._first
        else: self._cmd("moveto", str(e.y / max(1, self.winfo_height())))

    def _drag(self, e):
        if self._drag_y is None: return
        h = max(1, self.winfo_height()); dy = (e.y - self._drag_y) / h
        new = max(0.0, min(1.0 - (self._last - self._first), self._drag_first + dy))
        self._cmd("moveto", str(new))


# ── Rounded progress bar ──────────────────────────────────────────────────────
def _rrect(c, x1, y1, x2, y2, r, fill):
    r = min(r, max(1, (x2-x1)//2), max(1, (y2-y1)//2))
    c.create_oval(x1, y1, x1+2*r, y1+2*r,   fill=fill, outline="")
    c.create_oval(x2-2*r, y1, x2, y1+2*r,   fill=fill, outline="")
    c.create_oval(x1, y2-2*r, x1+2*r, y2,   fill=fill, outline="")
    c.create_oval(x2-2*r, y2-2*r, x2, y2,   fill=fill, outline="")
    c.create_rectangle(x1+r, y1, x2-r, y2,  fill=fill, outline="")
    c.create_rectangle(x1, y1+r, x2, y2-r,  fill=fill, outline="")

def draw_rprogress(canvas, value, color):
    canvas.delete("all")
    w = canvas.winfo_width(); h = canvas.winfo_height()
    if w < 4 or h < 2: return
    r = h // 2
    _rrect(canvas, 0, 0, w, h, r, BG)
    fw = max(0, int(w * value / 100))
    if fw >= h: _rrect(canvas, 0, 0, fw, h, r, color)


# ── Scrollable list (canvas-based, consistent with signals) ───────────────────
class ScrollList:
    """Canvas-backed scrollable list with LINE separators — same visual as signals panel."""

    def __init__(self, outer, accent, scrollables):
        wrap = tk.Frame(outer, bg=W, bd=0, highlightthickness=0)
        wrap.pack(fill="both", expand=True, padx=6, pady=(6, 12))

        self._canvas = tk.Canvas(wrap, bg=W, bd=0, highlightthickness=0)
        vsb = RoundScrollbar(wrap, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 4))
        self._canvas.pack(side="left", fill="both", expand=True)

        self._inner = tk.Frame(self._canvas, bg=W)
        win = self._canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._canvas.bind("<Configure>",
            lambda e: self._canvas.itemconfig(win, width=e.width))
        self._inner.bind("<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))

        # Direct scroll binding (backup for macOS)
        self._canvas.bind("<MouseWheel>",
            lambda e: self._canvas.yview_scroll(_scroll_units(e.delta), "units"))

        self._accent    = accent
        self._records   = []
        self._tag_map   = {}
        self._default   = "bot" if accent == RED else "human"
        scrollables.append(self._canvas)

    @property
    def canvas(self): return self._canvas

    def render(self, records, tag_map):
        self._records = records
        self._tag_map = tag_map
        for w in self._inner.winfo_children():
            w.destroy()
        for i, r in enumerate(records):
            self._add_row(r, tag_map.get(r["user_id"], self._default), i, len(records))
        self._canvas.yview_moveto(0)

    def _score_label(self, r, tag):
        pct = f"{int(r['bot_score']*100)}%"
        if tag == "tp": return f"✓ caught  {pct}"
        if tag == "fp": return f"✗ wrong flag  {pct}"
        if tag == "fn": return f"! missed  {pct}"
        return pct

    def _add_row(self, r, tag, idx, total):
        uid   = r["user_id"]
        name  = r.get("username") or uid
        color = TAG_COLOR.get(tag, TEXT)
        score = self._score_label(r, tag)

        row = tk.Frame(self._inner, bg=W)
        row.pack(fill="x")

        inner = tk.Frame(row, bg=W)
        inner.pack(fill="x", padx=16, pady=10)

        name_lbl  = tk.Label(inner, text=name,  bg=W, fg=color,
                              font=TF(12, bold=True), anchor="w")
        score_lbl = tk.Label(inner, text=score, bg=W, fg=SUB,
                              font=TF(11), anchor="e")
        name_lbl.pack(side="left")
        score_lbl.pack(side="right")

        def _hover(on, row=row, inner=inner, name_lbl=name_lbl, score_lbl=score_lbl):
            bg = BG if on else W
            for w in (row, inner, name_lbl, score_lbl):
                try: w.configure(bg=bg)
                except: pass

        scroll = lambda e: self._canvas.yview_scroll(_scroll_units(e.delta), "units")
        for w in (row, inner, name_lbl, score_lbl):
            w.bind("<Enter>",      lambda e: _hover(True))
            w.bind("<Leave>",      lambda e: _hover(False))
            w.bind("<MouseWheel>", scroll)

        if idx < total - 1:
            tk.Frame(self._inner, bg=LINE, height=1).pack(fill="x")

    def filter(self, q):
        q = q.lower().strip()
        filtered = [r for r in self._records
                    if not q or q in (r.get("username") or "").lower()]
        for w in self._inner.winfo_children():
            w.destroy()
        for i, r in enumerate(filtered):
            tag = self._tag_map.get(r["user_id"], self._default)
            self._add_row(r, tag, i, len(filtered))


class App(ctk.CTk):
    def __init__(self):
        open('/tmp/flagr_debug.txt','a').write("App.__init__ start\n")
        super().__init__()
        self.title("FlagR")
        self.geometry("1260x820")
        self.minsize(960, 620)
        self.configure(fg_color=BG)
        self.option_add("*Font", TF(12))
        self._flagged_ids    = []
        self._bot_records    = []
        self._hum_records    = []
        self._all_records    = []
        self._spinner_job    = None
        self._spinner_frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
        self._spinner_idx    = 0
        self._scrollables    = []   # canvas widgets for _on_wheel
        self._build()
        self.bind_all("<MouseWheel>", self._on_wheel)
        self.after(200, self._set_dock_icon)

    # ── Dock icon (pure-Python PNG, set via PyObjC) ───────────────────────
    def _set_dock_icon(self):
        try:
            self._set_dock_icon_impl()
        except Exception as e:
            import traceback
            open('/tmp/flagr_debug.txt','a').write(f"ERROR: {e}\n{traceback.format_exc()}\n")

    def _set_dock_icon_impl(self):
        import struct, zlib, tempfile, os, math, random
        open('/tmp/flagr_debug.txt','a').write("_set_dock_icon called\n")
        S = 512; PX = 36
        BG  = (13,13,26,255); GRN = (52,199,89,255)
        RED = (255,59,48,255); STR = (180,180,210,255); TRN = (0,0,0,0)
        img = [BG]*(S*S)

        def sp(x,y,c):
            if 0<=x<S and 0<=y<S: img[y*S+x]=c

        # Stars
        rng = random.Random(42)
        for _ in range(60):
            sp(rng.randint(14,S-14), rng.randint(14,S-14), STR)

        # Invader
        cols,rows_c = 10,8
        x0=(S-cols*PX)//2; y0=(S-rows_c*PX)//2-24
        for r,row in enumerate(self._INVADER):
            for c,val in enumerate(row):
                if val:
                    for py in range(PX-2):
                        for px_i in range(PX-2):
                            sp(x0+c*PX+px_i, y0+r*PX+py, GRN)

        # Crosshair ring (dashed)
        cxi=S//2; cyi=y0+rows_c*PX//2
        rr=cols*PX//2+36
        for deg in range(360):
            if (deg//10)%2==0:
                a=math.radians(deg)
                for dr in (-1,0,1):
                    sp(round(cxi+(rr+dr)*math.cos(a)), round(cyi+(rr+dr)*math.sin(a)), RED)

        # Crosshair lines (4 segments, leaving gap around invader)
        inner=rr-28; outer=rr+28
        for t in (-1,0,1):
            for i in range(inner, outer):
                sp(cxi+t, cyi-i, RED); sp(cxi+t, cyi+i, RED)
                sp(cxi-i, cyi+t, RED); sp(cxi+i, cyi+t, RED)

        # BOT DETECTED badge (red bar at bottom)
        bx1,bx2=cxi-110,cxi+110; by1,by2=S-110,S-64
        for y in range(by1,by2):
            for x in range(bx1,bx2): sp(x,y,RED)

        # Rounded corners (macOS icon style)
        cr=90
        for y in range(S):
            for x in range(S):
                if (x<cr or x>=S-cr) and (y<cr or y>=S-cr):
                    ccx=cr if x<cr else S-cr; ccy=cr if y<cr else S-cr
                    if math.sqrt((x-ccx)**2+(y-ccy)**2)>cr:
                        img[y*S+x]=TRN

        # Encode PNG (RGBA)
        def chunk(t,d):
            c=t+d; return struct.pack('>I',len(d))+c+struct.pack('>I',zlib.crc32(c)&0xffffffff)
        raw=bytearray()
        for y in range(S):
            raw.append(0)
            for x in range(S): raw.extend(img[y*S+x])
        path=os.path.join(tempfile.gettempdir(),'flagr_icon.png')
        with open(path,'wb') as f:
            f.write(b'\x89PNG\r\n\x1a\n')
            f.write(chunk(b'IHDR',struct.pack('>IIBBBBB',S,S,8,6,0,0,0)))
            f.write(chunk(b'IDAT',zlib.compress(bytes(raw),1)))
            f.write(chunk(b'IEND',b''))

        # Set dock icon via PyObjC
        try:
            from AppKit import NSApp, NSImage
            img_obj = NSImage.alloc().initWithContentsOfFile_(path)
            if img_obj is None:
                print("ERROR: NSImage failed to load", path)
                return
            NSApp.setActivationPolicy_(0)  # NSApplicationActivationPolicyRegular
            NSApp.setApplicationIconImage_(img_obj)
            NSApp.requestUserAttention_(0)  # bounce once to refresh dock
        except Exception as e:
            print("Dock icon error:", e)

    # ── Space Invader pixel art ───────────────────────────────────────────
    _INVADER = [
        [0,0,1,0,0,0,0,1,0,0],  # antennae tips
        [0,0,1,1,0,0,1,1,0,0],  # antennae bases
        [0,1,1,1,1,1,1,1,1,0],  # shoulders
        [1,1,1,1,1,1,1,1,1,1],  # body top
        [1,1,0,0,1,1,0,0,1,1],  # eyes (rectangular gaps)
        [1,1,1,1,1,1,1,1,1,1],  # body
        [0,1,1,0,0,0,0,1,1,0],  # lower body
        [1,1,0,0,0,0,0,0,1,1],  # feet
    ]

    def _draw_logo(self, canvas, cx, cy, px):
        """Draw invader centered at (cx, cy) with px pixel size."""
        cols, rows = 10, 8
        x0 = cx - cols * px // 2
        y0 = cy - rows * px // 2
        for r, row in enumerate(self._INVADER):
            for c, val in enumerate(row):
                if val:
                    x1, y1 = x0 + c*px, y0 + r*px
                    canvas.create_rectangle(x1, y1, x1+px-1, y1+px-1,
                                            fill=GRN, outline="")

    def _draw_welcome_logo(self, canvas):
        canvas.delete("all")
        w = canvas.winfo_width(); h = canvas.winfo_height()
        if w < 10: return
        # Stars
        import random; random.seed(42)
        for _ in range(22):
            sx = random.randint(4, w-4); sy = random.randint(4, h-4)
            canvas.create_oval(sx-1, sy-1, sx+1, sy+1,
                               fill="white", outline="", stipple="")
            canvas.itemconfigure("all")
        # Draw stars manually
        for sx, sy in [(18,12),(55,8),(95,18),(140,6),(165,22),(30,140),
                       (160,135),(12,80),(172,90),(80,148)]:
            canvas.create_oval(sx-1,sy-1,sx+1,sy+1, fill="white", outline="")

        px = 12
        cx, cy = w//2, h//2 - 8
        cols, rows = 10, 8
        x0 = cx - cols*px//2; y0 = cy - rows*px//2

        for r, row in enumerate(self._INVADER):
            for c, val in enumerate(row):
                if val:
                    x1, y1 = x0+c*px, y0+r*px
                    canvas.create_rectangle(x1, y1, x1+px-2, y1+px-2,
                                            fill=GRN, outline="")

        # Crosshair
        icx = cx; icy = cy - 4
        rad = 52
        canvas.create_oval(icx-rad, icy-rad, icx+rad, icy+rad,
                           outline=RED, width=2, dash=(8, 4))
        for dx, dy, ddx, ddy in [
            (0,-rad-6,0,-rad+6), (0,rad-6,0,rad+6),
            (-rad-6,0,-rad+6,0), (rad-6,0,rad+6,0)
        ]:
            canvas.create_line(icx+dx, icy+dy, icx+ddx, icy+ddy,
                               fill=RED, width=2)

        # BOT DETECTED label
        canvas.create_rectangle(cx-62, h-30, cx+62, h-8, fill=RED, outline="")
        canvas.create_text(cx, h-19, text="BOT DETECTED",
                           font=TF(9, bold=True), fill=W,
                           anchor="center")

    # ── Global wheel — walks widget ancestry ──────────────────────────────
    def _on_wheel(self, event):
        w = event.widget
        while w is not None:
            if w in self._scrollables:
                w.yview_scroll(_scroll_units(event.delta), "units")
                return
            try: w = w.master
            except: break
        # Fallback: pointer hit-test
        px, py = self.winfo_pointerx(), self.winfo_pointery()
        for widget in self._scrollables:
            try:
                if (widget.winfo_rootx() <= px <= widget.winfo_rootx() + widget.winfo_width() and
                        widget.winfo_rooty() <= py <= widget.winfo_rooty() + widget.winfo_height()):
                    widget.yview_scroll(_scroll_units(event.delta), "units"); return
            except: pass

    # ── Layout ────────────────────────────────────────────────────────────
    def _build(self):
        bar = ctk.CTkFrame(self, fg_color=W, corner_radius=0, height=62)
        bar.pack(fill="x"); bar.pack_propagate(False)

        # Logo canvas in top bar
        logo_bar = tk.Canvas(bar, width=36, height=36, bg=W, bd=0, highlightthickness=0)
        logo_bar.pack(side="left", padx=(20, 6), pady=13)
        self._draw_logo(logo_bar, 18, 18, 3)

        ctk.CTkLabel(bar, text="FlagR", font=F(17, bold=True),
                     text_color=TEXT).pack(side="left", padx=(0, 20))

        self._dl = ctk.CTkButton(bar, text="Export Bots", font=F(13),
            fg_color="transparent", hover_color=BG, text_color=TEXT,
            border_width=1, border_color=LINE, corner_radius=20, width=124, height=36,
            state="disabled", command=self._download)
        self._dl.pack(side="right", padx=(4, 24))

        self._cmp_btn = ctk.CTkButton(bar, text="Compare Ground Truth", font=F(13),
            fg_color="transparent", hover_color=BG, text_color=TEXT,
            border_width=1, border_color=LINE, corner_radius=20, width=178, height=36,
            state="disabled", command=self._load_ground_truth)
        self._cmp_btn.pack(side="right", padx=4)

        self._back_btn = ctk.CTkButton(bar, text="← New Dataset", font=F(13),
            fg_color=RED, hover_color="#cc2a20", text_color=W,
            corner_radius=20, width=124, height=36, state="disabled", command=self._reset)
        self._back_btn.pack(side="right", padx=4)

        self._status = ctk.CTkLabel(bar, text="", font=F(12), text_color=SUB)
        self._status.pack(side="right", padx=8)

        ctk.CTkFrame(self, fg_color=LINE, height=1, corner_radius=0).pack(fill="x")

        body = ctk.CTkFrame(self, fg_color=BG, corner_radius=0)
        body.pack(fill="both", expand=True, padx=24, pady=20)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        body.columnconfigure(2, weight=1)
        body.rowconfigure(0, minsize=92)
        body.rowconfigure(2, weight=1)

        # Top row: pills left, verification right — same row
        top_row = ctk.CTkFrame(body, fg_color="transparent")
        top_row.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        top_row.columnconfigure(1, weight=1)

        stats = ctk.CTkFrame(top_row, fg_color="transparent")
        stats.grid(row=0, column=0, sticky="w")
        self._s_bots   = self._pill(stats, "Bots",   "—", RED)
        self._s_humans = self._pill(stats, "Humans", "—", GRN)
        self._s_total  = self._pill(stats, "Total",  "—", TEXT)

        # Verification panel — floated to the right in the same row
        self._verify_frame = ctk.CTkFrame(top_row, fg_color="transparent")
        self._verify_frame.grid(row=0, column=2, sticky="e")
        self._verify_frame.grid_remove()

        vf_inner = ctk.CTkFrame(self._verify_frame, fg_color="#f0ebff", corner_radius=12)
        vf_inner.pack(side="left", ipadx=10, ipady=8)
        ctk.CTkLabel(vf_inner, text="SCORE", font=F(8, bold=True), text_color=PURPLE).pack(
            side="left", padx=(12, 10))
        self._v_acc   = self._verify_pill(vf_inner, "Score %",       "—", GRN)
        self._v_score = self._verify_pill(vf_inner, "Score",         "—", TEXT)
        self._v_tp    = self._verify_pill(vf_inner, "+2 Caught",     "—", GRN)
        self._v_fp    = self._verify_pill(vf_inner, "−6 Wrong Flag", "—", ORANGE)
        self._v_fn    = self._verify_pill(vf_inner, "−2 Missed",     "—", PURPLE)

        self._reset_btn = ctk.CTkButton(self._verify_frame, text="Clear", font=F(11),
            fg_color="transparent", hover_color=BG, text_color=TEXT,
            border_width=1, border_color=LINE, corner_radius=14, width=60, height=32,
            command=self._clear_comparison)
        self._reset_btn.pack(side="left", padx=(8, 0))

        # List columns
        self._bot_col = self._list_col(body, "Bots",   RED, 0)
        self._hum_col = self._list_col(body, "Humans", GRN, 1)

        # Signals
        self._build_signals(body, 2)

        # Welcome overlay
        self._welcome = ctk.CTkFrame(body, fg_color=BG, corner_radius=0)
        self._welcome.grid(row=2, column=0, columnspan=2, sticky="nsew")
        inner = ctk.CTkFrame(self._welcome, fg_color="transparent")
        inner.place(relx=0.5, rely=0.5, anchor="center")

        # Space Invader logo on dark card
        logo_card = tk.Canvas(inner, width=180, height=160,
                              bg="#0d0d1a", highlightthickness=0)
        logo_card.pack(pady=(0, 28))
        self._logo_card = logo_card
        logo_card.bind("<Configure>", lambda e: self._draw_welcome_logo(logo_card))

        ctk.CTkLabel(inner, text="Upload a dataset to begin.",
                     font=F(15), text_color=SUB).pack(pady=(0, 18))
        self._upload_btn = ctk.CTkButton(inner, text="Choose File",
            font=F(14, bold=True), fg_color=BLUE, hover_color="#005bbf",
            text_color=W, corner_radius=22, width=160, height=46, command=self._load)
        self._upload_btn.pack()

    # ── Verify pill — white card on purple bg, identical sizing to stat pill ─
    def _verify_pill(self, parent, label, value, color):
        card = ctk.CTkFrame(parent, fg_color=W, corner_radius=10, width=96, height=74)
        card.pack_propagate(False)
        card.pack(side="left", padx=(0, 8))
        v = ctk.CTkLabel(card, text=value, font=F(22, bold=True), text_color=color,
                         anchor="center", justify="center")
        v.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(card, text=label.upper(), font=F(8), text_color=SUB,
                     anchor="center", justify="center").pack(fill="x", pady=(0, 10))
        return v

    # ── Stat pill ─────────────────────────────────────────────────────────
    def _pill(self, parent, label, value, color):
        card = ctk.CTkFrame(parent, fg_color=W, corner_radius=10, width=96, height=74)
        card.pack_propagate(False)
        card.pack(side="left", padx=(0, 8))
        v = ctk.CTkLabel(card, text=value, font=F(22, bold=True), text_color=color,
                         anchor="center", justify="center")
        v.pack(fill="x", pady=(12, 0))
        ctk.CTkLabel(card, text=label.upper(), font=F(8), text_color=SUB,
                     anchor="center", justify="center").pack(fill="x", pady=(0, 10))
        return v

    # ── List column (canvas-based, consistent with signals) ───────────────
    def _list_col(self, parent, title, accent, col):
        outer = ctk.CTkFrame(parent, fg_color=W, corner_radius=16)
        outer.grid(row=2, column=col, sticky="nsew", padx=(0, 14))

        hdr = tk.Frame(outer, bg=W)
        hdr.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(hdr, text=title, bg=W, fg=TEXT,
                 font=TF(14, bold=True), anchor="w").pack(side="left")
        cnt_var = tk.StringVar(value="")
        cnt_lbl = tk.Label(hdr, textvariable=cnt_var, bg=W, fg=SUB,
                           font=TF(11), anchor="e")
        cnt_lbl.pack(side="right")

        sv = tk.StringVar()
        ctk.CTkEntry(outer, textvariable=sv, placeholder_text="Search…", font=F(12),
                     fg_color=BG, border_color=LINE, text_color=TEXT,
                     corner_radius=10, height=38).pack(fill="x", padx=14, pady=(0, 0))

        tk.Frame(outer, bg=LINE, height=1).pack(fill="x", pady=(8, 0))
        hdr.lift()

        sl = ScrollList(outer, accent, self._scrollables)
        sv.trace_add("write", lambda *_: sl.filter(sv.get()))

        outer._cnt = cnt_var
        outer._sl  = sl
        return outer

    # ── Signal sidebar ────────────────────────────────────────────────────
    def _build_signals(self, parent, col):
        outer = ctk.CTkFrame(parent, fg_color=W, corner_radius=16)
        outer.grid(row=2, column=col, sticky="nsew")

        hdr = tk.Frame(outer, bg=W)
        hdr.pack(fill="x", padx=20, pady=(12, 4))
        tk.Label(hdr, text="Signals", bg=W, fg=TEXT,
                 font=TF(14, bold=True), anchor="w").pack(side="left")
        tk.Label(hdr, text=str(len(SIGNALS)), bg=W, fg=SUB,
                 font=TF(11), anchor="e").pack(side="right")

        tk.Frame(outer, bg=LINE, height=1).pack(fill="x")

        wrap = tk.Frame(outer, bg=W, bd=0, highlightthickness=0)
        wrap.pack(fill="both", expand=True, padx=0, pady=(4, 12))
        hdr.lift()

        canvas = tk.Canvas(wrap, bg=W, bd=0, highlightthickness=0)
        vsb    = RoundScrollbar(wrap, command=canvas.yview)
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y", padx=(0, 4))
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas, bg=W)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win, width=e.width))
        inner.bind("<Configure>",  lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<MouseWheel>",
            lambda e: canvas.yview_scroll(_scroll_units(e.delta), "units"))

        self._scrollables.append(canvas)

        for i, (sig_name, imp, desc) in enumerate(SIGNALS):
            color = BLUE if imp >= 60 else (SUB if imp >= 40 else LINE)

            row = tk.Frame(inner, bg=W)
            row.pack(fill="x")

            content = tk.Frame(row, bg=W)
            content.pack(fill="x", padx=16, pady=10)

            hdrrow = tk.Frame(content, bg=W)
            hdrrow.pack(fill="x")
            tk.Label(hdrrow, text=sig_name, bg=W, fg=TEXT,
                     font=TF(10, bold=True), anchor="w").pack(side="left")
            tk.Label(hdrrow, text=f"{imp}%", bg=W, fg=color,
                     font=TF(10), anchor="e").pack(side="right")

            pb = tk.Canvas(content, height=7, bg=W, bd=0, highlightthickness=0)
            pb.pack(fill="x", pady=(4, 6))
            pb.bind("<Configure>", lambda e, c=pb, v=imp, cl=color: draw_rprogress(c, v, cl))

            desc_lbl = tk.Label(content, text=desc, bg=W, fg=SUB, font=TF(9),
                     anchor="w", wraplength=190, justify="left")
            desc_lbl.pack(fill="x", pady=(0, 0))

            # Direct wheel binding on every element so hover-scroll works
            scroll_fn = lambda e, cv=canvas: cv.yview_scroll(_scroll_units(e.delta), "units")
            for w in (row, content, hdrrow, pb, desc_lbl):
                w.bind("<MouseWheel>", scroll_fn)
            for lbl in hdrrow.winfo_children():
                lbl.bind("<MouseWheel>", scroll_fn)

            if i < len(SIGNALS) - 1:
                sep = tk.Frame(inner, bg=LINE, height=1)
                sep.pack(fill="x")
                sep.bind("<MouseWheel>", scroll_fn)

    # ── Spinner ───────────────────────────────────────────────────────────
    def _spin(self):
        f = self._spinner_frames[self._spinner_idx % len(self._spinner_frames)]
        self._spinner_idx += 1
        self._upload_btn.configure(text=f"{f}  Analyzing…", state="disabled")
        self._status.configure(text=f"{f}  Analyzing…")
        self._spinner_job = self.after(80, self._spin)

    def _spin_stop(self):
        if self._spinner_job: self.after_cancel(self._spinner_job); self._spinner_job = None
        self._upload_btn.configure(text="Choose File", state="normal")

    def _load(self):
        path = filedialog.askopenfilename(title="Select dataset",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not path: return
        self._spinner_idx = 0; self._spin()
        threading.Thread(target=self._run, args=(path,), daemon=True).start()

    def _run(self, path):
        try:
            dataset = load_dataset(path)
            _, _, records = score_dataset(dataset, ARTIFACTS_DIR)
            bots   = sorted([r for r in records if     r["flagged"]], key=lambda r: -r["bot_score"])
            humans = sorted([r for r in records if not r["flagged"]], key=lambda r:  r["bot_score"])
            self._flagged_ids = [r["user_id"] for r in records if r["flagged"]]
            self._all_records = records
            self.after(0, lambda: (self._spin_stop(), self._show(bots, humans, len(records))))
        except Exception as ex:
            msg = str(ex)
            self.after(0, lambda msg=msg: (self._spin_stop(),
                                           self._status.configure(text=f"Error: {msg}")))

    def _show(self, bots, humans, total):
        if self._welcome.winfo_exists(): self._welcome.grid_forget()
        self._s_bots.configure(text=str(len(bots)))
        self._s_humans.configure(text=str(len(humans)))
        self._s_total.configure(text=str(total))
        self._status.configure(text="")
        self._bot_records = bots; self._hum_records = humans
        self._bot_col._cnt.set(str(len(bots)))
        self._hum_col._cnt.set(str(len(humans)))
        self._bot_col._sl.render(bots,   {})
        self._hum_col._sl.render(humans, {})
        self._dl.configure(state="normal")
        self._cmp_btn.configure(state="normal")
        self._back_btn.configure(state="normal")
        self._verify_frame.grid_remove()

    # ── Ground truth ──────────────────────────────────────────────────────
    def _load_ground_truth(self):
        path = filedialog.askopenfilename(title="Select ground truth bots file",
            filetypes=[("Text file", "*.txt"), ("All", "*.*")])
        if not path: return
        try:
            with open(path, encoding="utf-8") as f:
                ground_truth = {line.strip() for line in f if line.strip()}
            self._apply_comparison(ground_truth)
        except Exception as ex:
            self._status.configure(text=f"Error: {ex}")

    def _apply_comparison(self, ground_truth: set):
        predicted = set(self._flagged_ids)
        tp = len(predicted & ground_truth)
        fp = len(predicted - ground_truth)
        fn = len(ground_truth - predicted)
        score = competition_score(tp, fp, fn)

        bot_map = {r["user_id"]: ("tp" if r["user_id"] in ground_truth else "fp")
                   for r in self._bot_records}
        hum_map = {r["user_id"]: ("fn" if r["user_id"] in ground_truth else "human")
                   for r in self._hum_records}

        self._bot_col._sl.render(self._bot_records, bot_map)
        self._hum_col._sl.render(self._hum_records, hum_map)

        max_possible = 2 * (tp + fn)
        acc_pct   = (score / max_possible * 100) if max_possible > 0 else 100.0
        acc_color = GRN if acc_pct >= 90 else (ORANGE if acc_pct >= 50 else RED)

        self._v_acc.configure(text=f"{acc_pct:+.0f}%",  text_color=acc_color)
        self._v_score.configure(text=f"{score:+d}",      text_color=(GRN if score >= 0 else RED))
        self._v_tp.configure(text=str(tp))
        self._v_fp.configure(text=str(fp))
        self._v_fn.configure(text=str(fn))
        self._verify_frame.grid()
        self._status.configure(
            text=f"Score %: {acc_pct:+.0f}%  •  {tp} caught  •  {fp} wrong  •  {fn} missed")

    def _clear_comparison(self):
        self._bot_col._sl.render(self._bot_records, {})
        self._hum_col._sl.render(self._hum_records, {})
        self._verify_frame.grid_remove()
        self._status.configure(text="")

    # ── Reset ─────────────────────────────────────────────────────────────
    def _reset(self):
        self._flagged_ids = []; self._bot_records = []; self._hum_records = []; self._all_records = []
        self._bot_col._sl.render([], {}); self._hum_col._sl.render([], {})
        self._bot_col._cnt.set(""); self._hum_col._cnt.set("")
        self._verify_frame.grid_remove()
        self._s_bots.configure(text="—"); self._s_humans.configure(text="—")
        self._s_total.configure(text="—"); self._status.configure(text="")
        self._dl.configure(state="disabled")
        self._cmp_btn.configure(state="disabled")
        self._back_btn.configure(state="disabled")
        self._welcome.grid(row=2, column=0, columnspan=2, sticky="nsew")

    # ── Export ────────────────────────────────────────────────────────────
    def _download(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
            filetypes=[("Text file", "*.txt")], initialfile="detections.txt")
        if not path: return
        with open(path, "w") as f:
            for uid in self._flagged_ids: f.write(f"{uid}\n")
        self._status.configure(text=f"Saved {os.path.basename(path)}")


if __name__ == "__main__":
    App().mainloop()
