#!/usr/bin/env python3
"""A tiny curses checkbox TUI for REVIEW.md.

The file is treated as immutable except for the three characters `[ ]` / `[x]`
inside a checkbox line. Everything else -- indentation, trailing whitespace,
line endings, unicode -- round-trips verbatim.

Usage:
    ./checklist.py [path/to/REVIEW.md]

Keys:
    j / k / arrows   move          g / G   top / bottom
    space / enter    toggle        q       quit
    PgDn / PgUp      scroll the detail pane
"""

from __future__ import annotations

import curses
import locale
import os
import re
import sys
import tempfile
import textwrap
from dataclasses import dataclass, field

DEFAULT_FILENAME = "REVIEW.md"
DETAIL_INDENT = 6

# `- [ ] ...` / `- [x] ...` at column 0. Group 1 is the single state character;
# it always sits at line[3], so the mutable span is line[2:5].
CHECKBOX_RE = re.compile(r"^- \[([ xX])\] (.*)$")
# `**A1. Some summary**` followed by optional trailing text.
LABEL_RE = re.compile(r"^\*\*([A-Za-z]+\d+)\.\s+(.*?)\*\*(.*)$")
SECTION_RE = re.compile(r"^##\s+(.*)$")

CHECKBOX_SPAN = slice(2, 5)


# --------------------------------------------------------------------------
# Pure model: parse / toggle / serialize. No curses below this line until the
# UI section, so all of it is testable without a tty.
# --------------------------------------------------------------------------


@dataclass
class Item:
    """One `- [ ] **A1. ...**` checkbox and its indented detail block."""

    line_index: int
    ident: str
    summary: str
    trailing: str
    checked: bool
    detail: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return f"{self.ident}. {self.summary}" if self.ident else self.summary


@dataclass
class Section:
    """A `## ` header. Shown in the list but not selectable."""

    line_index: int
    title: str


@dataclass
class Document:
    path: str
    lines: list[str]  # keepends=True, joined back verbatim
    rows: list[object]  # interleaved Section | Item, in file order

    @property
    def items(self) -> list[Item]:
        return [r for r in self.rows if isinstance(r, Item)]

    @property
    def done_count(self) -> int:
        return sum(1 for i in self.items if i.checked)

    def serialize(self) -> str:
        return "".join(self.lines)

    def toggle(self, item: Item) -> None:
        """Flip one checkbox, mutating only lines[n][2:5]."""
        line = self.lines[item.line_index]
        head, span, tail = line[:2], line[CHECKBOX_SPAN], line[5:]
        if span not in ("[ ]", "[x]", "[X]"):  # defensive; parser guarantees it
            raise ValueError(f"line {item.line_index + 1} is not a checkbox line")
        new_span = "[ ]" if item.checked else "[x]"
        self.lines[item.line_index] = head + new_span + tail
        item.checked = not item.checked


def parse_text(text: str, path: str = "") -> Document:
    """Split into physical lines and index the checkboxes and headers.

    Only column-0 `- [ ]` lines count, so the fenced python block in B1 (which
    is indented six spaces, like every other detail line) can contain anything
    -- backticks, `1.` ordered lists, markdown -- without confusing us.
    """
    lines = text.splitlines(keepends=True)
    rows: list[object] = []

    for i, raw in enumerate(lines):
        body = raw.rstrip("\r\n")

        section = SECTION_RE.match(body)
        if section:
            rows.append(Section(line_index=i, title=section.group(1).strip()))
            continue

        checkbox = CHECKBOX_RE.match(body)
        if not checkbox:
            continue

        state, rest = checkbox.group(1), checkbox.group(2)
        label = LABEL_RE.match(rest)
        if label:
            ident, summary, trailing = label.group(1), label.group(2), label.group(3)
        else:  # tolerate a checkbox that isn't bold-labelled
            ident, summary, trailing = "", rest.strip(), ""
        rows.append(
            Item(
                line_index=i,
                ident=ident,
                summary=summary.strip(),
                trailing=trailing.strip(),
                checked=state in ("x", "X"),
            )
        )

    _attach_details(lines, rows)
    return Document(path=path, lines=lines, rows=rows)


def _attach_details(lines: list[str], rows: list[object]) -> None:
    """Detail runs from just after an item to the next item or `## ` header."""
    stops = sorted(r.line_index for r in rows)
    for row in rows:
        if not isinstance(row, Item):
            continue
        start = row.line_index + 1
        end = len(lines)
        for stop in stops:
            if stop >= start:
                end = stop
                break
        block = [ln.rstrip("\r\n") for ln in lines[start:end]]
        while block and not block[-1].strip():
            block.pop()
        row.detail = [_dedent(ln) for ln in block]


def _dedent(line: str) -> str:
    """Drop up to DETAIL_INDENT leading spaces, keeping any extra indent.

    Display-only: `lines` keeps the original text for write-back.
    """
    strip = len(line) - len(line.lstrip(" "))
    return line[min(strip, DETAIL_INDENT) :]


def read_document(path: str) -> Document:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return parse_text(fh.read(), path=path)


def write_document(doc: Document) -> None:
    """Atomic replace so a crash mid-write cannot truncate the file."""
    directory = os.path.dirname(os.path.abspath(doc.path)) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".checklist-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(doc.serialize())
        try:
            os.chmod(tmp, os.stat(doc.path).st_mode & 0o777)
        except OSError:
            pass
        os.replace(tmp, doc.path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def wrap_detail(detail: list[str], width: int) -> list[str]:
    """Wrap to `width`, keeping each line's own indentation on continuations."""
    if width < 8:
        width = 8
    out: list[str] = []
    for line in detail:
        if not line.strip():
            out.append("")
            continue
        indent = line[: len(line) - len(line.lstrip(" "))]
        pieces = textwrap.wrap(
            line.strip(),
            width=width,
            initial_indent=indent,
            subsequent_indent=indent + "  ",
            break_long_words=True,
            break_on_hyphens=False,
        )
        out.extend(pieces or [""])
    return out


# --------------------------------------------------------------------------
# curses UI
# --------------------------------------------------------------------------

MIN_WIDTH = 40
MIN_HEIGHT = 10

C_HEADER = 1
C_SECTION = 2
C_DONE = 3
C_HINT = 4
C_ERROR = 5


class App:
    def __init__(self, doc: Document) -> None:
        self.doc = doc
        self.rows = doc.rows
        self.selected = self._first_selectable()
        self.list_top = 0
        self.detail_top = 0
        self.status = ""

    # -- selection ------------------------------------------------------
    def _first_selectable(self) -> int:
        for i, row in enumerate(self.rows):
            if isinstance(row, Item):
                return i
        return 0

    def _last_selectable(self) -> int:
        for i in range(len(self.rows) - 1, -1, -1):
            if isinstance(self.rows[i], Item):
                return i
        return 0

    def move(self, delta: int) -> None:
        i = self.selected
        while True:
            i += delta
            if i < 0 or i >= len(self.rows):
                return  # no selectable row that way; stay put
            if isinstance(self.rows[i], Item):
                self.selected = i
                self.detail_top = 0
                self.status = ""
                return

    def current_item(self) -> Item | None:
        row = self.rows[self.selected] if self.rows else None
        return row if isinstance(row, Item) else None

    # -- actions --------------------------------------------------------
    def toggle(self) -> None:
        item = self.current_item()
        if item is None:
            return
        self.doc.toggle(item)
        try:
            write_document(self.doc)
            self.status = ""
        except OSError as exc:
            self.doc.toggle(item)  # roll back so screen matches disk
            self.status = f"write failed: {exc}"

    # -- drawing --------------------------------------------------------
    def draw(self, stdscr) -> None:
        stdscr.erase()
        height, width = stdscr.getmaxyx()
        if height < MIN_HEIGHT or width < MIN_WIDTH:
            msg = f"terminal too small: need {MIN_WIDTH}x{MIN_HEIGHT}"
            _addstr(stdscr, 0, 0, msg[: max(0, width - 1)], _attr(C_ERROR))
            stdscr.noutrefresh()
            curses.doupdate()
            return

        body = height - 2  # title and footer; the separator lives in detail_h
        list_h = max(3, min(len(self.rows), body // 2))
        detail_h = body - list_h

        self._draw_title(stdscr, width)
        self._draw_list(stdscr, 1, list_h, width)
        self._draw_detail(stdscr, 1 + list_h, detail_h, width)
        self._draw_footer(stdscr, height - 1, width)
        stdscr.noutrefresh()
        curses.doupdate()

    def _draw_title(self, stdscr, width: int) -> None:
        done, total = self.doc.done_count, len(self.doc.items)
        name = os.path.basename(self.doc.path)
        left = f" {name} "
        right = f" {done}/{total} done "
        bar = ""
        slots = width - len(left) - len(right) - 3
        if slots >= 10 and total:
            filled = round(slots * done / total)
            bar = "[" + "#" * filled + "." * (slots - filled) + "]"
        text = (left + bar).ljust(max(0, width - len(right) - 1)) + right
        _addstr(stdscr, 0, 0, text[: width - 1], _attr(C_HEADER) | curses.A_BOLD)

    def _draw_list(self, stdscr, top: int, height: int, width: int) -> None:
        self._scroll_list_into_view(height)
        for offset in range(height):
            index = self.list_top + offset
            if index >= len(self.rows):
                break
            row = self.rows[index]
            y = top + offset
            if isinstance(row, Section):
                _addstr(
                    stdscr,
                    y,
                    0,
                    f"  {row.title}"[: width - 1],
                    _attr(C_SECTION) | curses.A_BOLD,
                )
                continue

            selected = index == self.selected
            mark = ">" if selected else " "
            box = "[x]" if row.checked else "[ ]"
            text = f"{mark} {box} {row.title}"
            attr = curses.A_NORMAL
            if row.checked:
                attr |= _attr(C_DONE) | curses.A_DIM
            if selected:
                attr = curses.A_REVERSE | curses.A_BOLD
                text = text.ljust(width - 1)
            _addstr(stdscr, y, 0, text[: width - 1], attr)

        # scroll affordances
        if self.list_top > 0:
            _addstr(stdscr, top, width - 2, "^", _attr(C_HINT))
        if self.list_top + height < len(self.rows):
            _addstr(stdscr, top + height - 1, width - 2, "v", _attr(C_HINT))

    def _scroll_list_into_view(self, height: int) -> None:
        if self.selected < self.list_top:
            self.list_top = self.selected
        elif self.selected >= self.list_top + height:
            self.list_top = self.selected - height + 1
        max_top = max(0, len(self.rows) - height)
        self.list_top = max(0, min(self.list_top, max_top))

    def _draw_detail(self, stdscr, top: int, height: int, width: int) -> None:
        item = self.current_item()
        pane_h = height - 1  # one row for the separator/label
        wrapped = wrap_detail(item.detail, width - 3) if item else []
        if item and item.trailing:
            wrapped = [item.trailing, ""] + wrapped

        max_top = max(0, len(wrapped) - pane_h)
        self.detail_top = max(0, min(self.detail_top, max_top))

        label = f" {item.ident} " if item and item.ident else " detail "
        if len(wrapped) > pane_h:
            shown_lo = self.detail_top + 1
            shown_hi = min(self.detail_top + pane_h, len(wrapped))
            label += f"{shown_lo}-{shown_hi}/{len(wrapped)} "
        rule = ("--" + label).ljust(width - 1, "-")
        _addstr(stdscr, top, 0, rule[: width - 1], _attr(C_HINT))

        for offset in range(pane_h):
            index = self.detail_top + offset
            if index >= len(wrapped):
                break
            _addstr(stdscr, top + 1 + offset, 2, wrapped[index][: width - 3])

    def _draw_footer(self, stdscr, y: int, width: int) -> None:
        if self.status:
            _addstr(stdscr, y, 0, self.status[: width - 1], _attr(C_ERROR))
            return
        hint = " j/k move  g/G top/bottom  space toggle  PgDn/PgUp detail  q quit"
        _addstr(stdscr, y, 0, hint[: width - 1], _attr(C_HINT))

    # -- loop -----------------------------------------------------------
    def run(self, stdscr) -> None:
        curses.curs_set(0)
        stdscr.keypad(True)
        _init_colors()
        while True:
            self.draw(stdscr)
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                return
            if key in (ord("q"), ord("Q")):
                return
            elif key in (curses.KEY_DOWN, ord("j")):
                self.move(1)
            elif key in (curses.KEY_UP, ord("k")):
                self.move(-1)
            elif key == ord("g"):
                self.selected = self._first_selectable()
                self.detail_top = 0
            elif key == ord("G"):
                self.selected = self._last_selectable()
                self.detail_top = 0
            elif key in (ord(" "), curses.KEY_ENTER, 10, 13):
                self.toggle()
            elif key in (curses.KEY_NPAGE, 4):  # PgDn, Ctrl-D
                self.detail_top += 5
            elif key in (curses.KEY_PPAGE, 21):  # PgUp, Ctrl-U
                self.detail_top = max(0, self.detail_top - 5)
            elif key == curses.KEY_RESIZE:
                continue
            elif key in (ord("r"), ord("R")):  # reload from disk
                try:
                    self.doc = read_document(self.doc.path)
                    self.rows = self.doc.rows
                    self.selected = min(self.selected, len(self.rows) - 1)
                    self.status = ""
                except OSError as exc:
                    self.status = f"reload failed: {exc}"


def _init_colors() -> None:
    if not curses.has_colors():
        return
    curses.start_color()
    try:
        curses.use_default_colors()
        bg = -1
    except curses.error:
        bg = curses.COLOR_BLACK
    curses.init_pair(C_HEADER, curses.COLOR_CYAN, bg)
    curses.init_pair(C_SECTION, curses.COLOR_YELLOW, bg)
    curses.init_pair(C_DONE, curses.COLOR_GREEN, bg)
    curses.init_pair(C_HINT, curses.COLOR_BLUE, bg)
    curses.init_pair(C_ERROR, curses.COLOR_RED, bg)


def _attr(pair: int) -> int:
    try:
        if curses.has_colors():
            return curses.color_pair(pair)
    except curses.error:
        pass
    return curses.A_NORMAL


def _addstr(win, y: int, x: int, text: str, attr: int = curses.A_NORMAL) -> None:
    """addstr that tolerates the bottom-right cell and off-screen writes."""
    if not text:
        return
    try:
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def resolve_path(argv: list[str]) -> str:
    if len(argv) > 2:
        raise SystemExit(f"usage: {os.path.basename(argv[0])} [REVIEW.md]")
    if len(argv) == 2:
        return argv[1]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), DEFAULT_FILENAME)


def main(argv: list[str]) -> int:
    locale.setlocale(locale.LC_ALL, "")
    path = resolve_path(argv)

    try:
        doc = read_document(path)
    except FileNotFoundError:
        print(f"checklist: no such file: {path}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"checklist: cannot read {path}: {exc}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as exc:
        print(f"checklist: {path} is not valid UTF-8: {exc}", file=sys.stderr)
        return 1

    if not doc.items:
        print(f"checklist: no `- [ ]` checkbox lines found in {path}", file=sys.stderr)
        return 1

    if not sys.stdout.isatty():
        done, total = doc.done_count, len(doc.items)
        print(
            f"checklist: needs an interactive terminal (found {total} items, "
            f"{done} done in {path})",
            file=sys.stderr,
        )
        return 1

    app = App(doc)
    try:
        curses.wrapper(app.run)
    except curses.error as exc:
        print(f"checklist: terminal not usable by curses: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
