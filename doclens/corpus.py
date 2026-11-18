"""Read the frozen source inventory and turn official docs into citable passages."""

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from textwrap import dedent
from tempfile import NamedTemporaryFile
from typing import Iterator, Literal
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1] / "corpus"
Strategy = Literal["heading", "recursive"]

HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(`{3,}|~{3,})\s*(\{[\w-]+\}|[\w+-]*)")
FENCED_CODE = re.compile(r"^[ \t]*```[^\n]*\n.*?^[ \t]*```[ \t]*$", flags=re.M | re.S)
INCLUDE = re.compile(r"\{[*!]>?.*?(docs_src/[\w/.-]+\.py)\b")
RST_DIVIDER = re.compile(r"^([=\-~^\"`:+*])\1{2,}$")
RST_DIRECTIVE = re.compile(r"^(\s*)\.\.\s+([\w:-]+)::\s*(.*)$")
RST_CODE = {"code", "code-block", "doctest", "testcode", "testoutput", "sourcecode"}
RST_SIGNATURE = {
    "class", "function", "method", "classmethod", "staticmethod", "attribute", "data",
    "exception", "coroutinefunction", "awaitablefunction", "decorator", "property",
}
MARKUP = re.compile(r"</?[a-zA-Z][\w-]*(?:\s+[^<>]*)?\s*/?>")
INLINE_LINK = re.compile(r"\[([^\]]+)\]\([^\s)]+(?:\s+\"[^\"]*\")?\)(?:\{[^}]+\})?")
RST_ROLE = re.compile(r"(?::[\w:-]+:)?`([^`\n]+)`_{0,2}")
MDX_COMPONENT = re.compile(r"^\s*</?[A-Z][\w.]*\b")
MAX_SOURCE_BYTES = 2_000_000


@dataclass(frozen=True)
class Collection:
    id: str
    topic: str
    title: str
    description: str
    repository: str
    version: str
    revision: str
    license: str
    license_path: str
    license_sha256: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.repository}/tree/{self.revision}"


@dataclass(frozen=True)
class Source:
    id: str
    collection: Collection
    title: str
    path: str
    url: str
    sha256: str

    @property
    def format(self) -> str:
        return Path(self.path).suffix.lstrip(".")

    @property
    def pinned_url(self) -> str:
        return f"https://github.com/{self.collection.repository}/blob/{self.collection.revision}/{self.path}"

    def excerpt_url(self, start: int, end: int) -> str:
        return f"{self.pinned_url}?plain=1#L{start}-L{end}"


@dataclass(frozen=True)
class Chunk:
    id: str
    source: Source
    heading: tuple[str, ...]
    text: str
    start: int
    end: int

    @property
    def url(self) -> str:
        return self.source.excerpt_url(self.start, self.end)

    @property
    def search_text(self) -> str:
        return "\n".join((self.source.title, " ".join(self.heading), self.text))


@dataclass(frozen=True)
class Block:
    text: str
    start: int
    end: int
    heading_level: int = 0
    language: str | None = None

    @property
    def markdown(self) -> str:
        if self.language is None:
            return self.text
        return f"```{self.language}\n{self.text}\n```"


class Corpus:
    def __init__(self, root: Path = ROOT):
        self.root = root
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        self.retrieved_on = manifest["retrieved_on"]
        self.collections = {entry["id"]: Collection(**entry) for entry in manifest["collections"]}
        self.sources = [
            Source(collection=self.collections[entry["collection"]], **{
                key: value for key, value in entry.items() if key != "collection"
            })
            for entry in manifest["documents"]
        ]
        self.examples = manifest["examples"]

    def assets(self) -> Iterator[tuple[Collection, str, Path, str]]:
        for collection in self.collections.values():
            yield (
                collection, collection.license_path,
                self.root / "licenses" / f"{collection.id}.txt", collection.license_sha256,
            )
        for source in self.sources:
            yield (
                source.collection, source.path,
                self.root / "raw" / f"{source.id}.{source.format}", source.sha256,
            )
        for entry in self.examples:
            yield (
                self.collections[entry["collection"]], entry["path"],
                self.root / "examples" / f"{entry['path'].removeprefix('docs_src/')}.txt",
                entry["sha256"],
            )

    @staticmethod
    def check(data: bytes, expected: str, label: str) -> bytes:
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Source checksum mismatch: {label}")
        return data

    def verify(self) -> int:
        count = 0
        for _, _, local, digest in self.assets():
            self.check(local.read_bytes(), digest, str(local))
            count += 1
        return count

    def fetch(self) -> int:
        """Restore missing assets from immutable official snapshots."""
        count = 0
        for collection, remote, local, digest in self.assets():
            if local.exists():
                self.check(local.read_bytes(), digest, str(local))
                continue
            url = f"https://raw.githubusercontent.com/{collection.repository}/{collection.revision}/{remote}"
            with urlopen(url, timeout=30) as response:
                data = response.read(MAX_SOURCE_BYTES + 1)
            if len(data) > MAX_SOURCE_BYTES:
                raise ValueError(f"Source exceeds the download size limit: {remote}")
            self.check(data, digest, url)
            local.parent.mkdir(parents=True, exist_ok=True)
            with NamedTemporaryFile(dir=local.parent, delete=False) as staging:
                staging.write(data)
                staged = Path(staging.name)
            try:
                staged.replace(local)
            finally:
                staged.unlink(missing_ok=True)
            count += 1
        return count

    def chunks(self, strategy: Strategy = "heading", target: int = 360) -> list[Chunk]:
        self.verify()
        examples = {
            entry["path"]: (self.root / "examples" / f"{entry['path'].removeprefix('docs_src/')}.txt")
            .read_text(encoding="utf-8").rstrip()
            for entry in self.examples
        }
        chunks = []
        for source in self.sources:
            raw = (self.root / "raw" / f"{source.id}.{source.format}").read_text(encoding="utf-8")
            blocks = rst_blocks(raw) if source.format == "rst" else markdown_blocks(raw, examples)
            chunks.extend(pack(source, blocks, strategy, target))
        return chunks


def clean_inline(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = INLINE_LINK.sub(r"\1", text)
    text = MARKUP.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def rst_inline(text: str) -> str:
    def role(match: re.Match) -> str:
        label = re.sub(r"\s+<[^>]+>$", "", match[1]).lstrip("~!")
        return f"`{label}`"

    return clean_inline(RST_ROLE.sub(role, text.replace("``", "`")))


def prose(lines: list[tuple[int, str]], *, rst: bool = False) -> Iterator[Block]:
    paragraph = []
    for number, line in [*lines, (0, "")]:
        if line.strip():
            paragraph.append((number, line))
            continue
        if paragraph:
            first, _ = paragraph[0]
            last, _ = paragraph[-1]
            list_line = re.compile(r"^(?:[-*+]|\d+\.)\s")
            cleaned = []
            for _, raw in paragraph:
                stripped = raw.strip()
                if list_line.match(stripped) and cleaned and not cleaned[-1].endswith("\n"):
                    cleaned.append("\n")
                cleaned.append((rst_inline if rst else clean_inline)(stripped))
            text = " ".join(cleaned).replace(" \n ", "\n").strip()
            if text:
                yield Block(text, first, last)
            paragraph = []


def markdown_blocks(raw: str, examples: dict[str, str]) -> Iterator[Block]:
    # Keep source line numbers for immutable links to the original evidence.
    raw = re.sub(r"<!--.*?-->", lambda match: "\n" * match[0].count("\n"), raw, flags=re.S)
    lines = raw.splitlines()
    position = 0
    if lines and lines[0].strip() == "---" and "---" in lines[1:]:
        position = lines.index("---", 1) + 1
    pending = []
    shown = set()
    while position < len(lines):
        line = lines[position]
        current = position + 1
        position += 1
        fence = FENCE.match(line)
        heading = HEADING.match(line.lstrip())
        include = INCLUDE.search(line)
        if fence or heading or include:
            yield from prose(pending)
            pending = []
        if fence:
            body = []
            closer = re.compile(rf"^\s*{fence[1][0]}{{{len(fence[1])},}}\s*$")
            while position < len(lines) and not closer.match(lines[position]):
                body.append(lines[position])
                position += 1
            position = min(position + 1, len(lines))
            if body and fence[2] == "{eval-rst}":
                for block in rst_blocks(dedent("\n".join(body))):
                    yield replace(block, start=block.start + current, end=block.end + current)
            elif body and fence[2] != "mermaid":
                yield Block(dedent("\n".join(body)).strip("\n"), current, position, language=fence[2])
            continue
        if heading:
            yield Block(clean_inline(heading[2]), current, current, len(heading[1]))
            continue
        if include:
            path = include[1]
            if path in examples and path not in shown:
                shown.add(path)
                yield Block(examples[path], current, current, language="python")
            continue
        if MDX_COMPONENT.match(line):
            while ">" not in line and position < len(lines):
                line = lines[position]
                position += 1
            continue
        if re.match(r"^\s*(?:/{3}|:::+|===\s|import\s.+from\s|\[\[autodoc\]\]|[{}]\s*$)", line):
            continue
        if line.startswith("|") or re.fullmatch(r"\s*[-*=]{3,}\s*", line):
            # Markdown tables are flattened separately from surrounding paragraphs.
            if line.startswith("|") and not re.fullmatch(r"[|\s:-]+", line):
                pending.append((current, " · ".join(cell.strip() for cell in line.strip("|").split("|"))))
            continue
        pending.append((current, line))
    yield from prose(pending)


def indented_block(lines: list[str], start: int, base_indent: int) -> tuple[list[str], int]:
    cursor = start
    while cursor < len(lines):
        line = lines[cursor]
        if line.strip() and len(line) - len(line.lstrip()) <= base_indent:
            break
        cursor += 1
    body = lines[start:cursor]
    while body and (not body[0].strip() or re.match(r"^\s*:[\w-]+:", body[0])):
        body = body[1:]
    return body, cursor


def rst_blocks(raw: str) -> Iterator[Block]:
    lines = raw.splitlines()
    levels = {}
    position = 0
    pending = []
    while position < len(lines):
        line = lines[position]
        current = position + 1
        position += 1
        if not line.startswith(" ") and line.strip() and position < len(lines):
            divider = RST_DIVIDER.fullmatch(lines[position])
            if divider and not RST_DIVIDER.fullmatch(line):
                yield from prose(pending, rst=True)
                pending = []
                level = levels.setdefault(divider[1], len(levels) + 1)
                yield Block(rst_inline(line), current, current + 1, level)
                position += 1
                continue
        directive = RST_DIRECTIVE.match(line)
        if directive:
            indent, name, label = directive.groups()
            name = name.split(":")[-1]
            if name in RST_CODE:
                yield from prose(pending, rst=True)
                pending = []
                body, position = indented_block(lines, position, len(indent))
                if body and (code := dedent("\n".join(body)).strip("\n")):
                    language = label.split()[0] if label else "python"
                    yield Block(code, current, position, language=language)
            elif name in RST_SIGNATURE and label:
                yield from prose(pending, rst=True)
                pending = []
                yield Block(rst_inline(label), current, current, max(levels.values(), default=1) + 1)
            elif name in {"versionadded", "versionchanged", "deprecated"}:
                readable = {"versionadded": "Added in version", "versionchanged": "Changed in version",
                            "deprecated": "Deprecated since version"}[name]
                pending.extend([(current, ""), (current, f"{readable} {label}.")])
            elif name not in {"note", "warning", "seealso", "important", "tip", "admonition"}:
                position = indented_block(lines, position, len(indent))[1]
            continue
        if line.rstrip().endswith("::") and not line.lstrip().startswith(".."):
            if line.strip() != "::":
                pending.append((current, line.rstrip()[:-1]))
            yield from prose(pending, rst=True)
            pending = []
            body, position = indented_block(lines, position, len(line) - len(line.lstrip()))
            if body and (code := dedent("\n".join(body)).strip("\n")):
                yield Block(code, current + 1, position, language="python")
            continue
        if line.lstrip().startswith(".. ") or RST_DIVIDER.fullmatch(line):
            continue
        pending.append((current, line))
    yield from prose(pending, rst=True)


def token_count(text: str) -> int:
    return len(re.findall(r"\w+|[^\s\w]", text))


def bounded(block: Block, target: int) -> Iterator[Block]:
    if token_count(block.text) <= target:
        yield block
        return
    separator = "\n" if block.language is not None else " "
    units = block.text.splitlines() if block.language is not None else block.text.split()
    batch = []
    length = 0
    for unit in units:
        size = token_count(unit)
        if batch and length + size > target:
            yield Block(separator.join(batch), block.start, block.end, language=block.language)
            batch, length = [], 0
        batch.append(unit)
        length += size
    if batch:
        yield Block(separator.join(batch), block.start, block.end, language=block.language)


def pack(source: Source, blocks: Iterator[Block], strategy: Strategy, target: int) -> Iterator[Chunk]:
    staged = []
    headings: list[tuple[int, str]] = []
    length = 0
    sequence = 0

    def chunk(items: list[Block]) -> Chunk:
        nonlocal sequence
        sequence += 1
        first, last = items[0].start, items[-1].end
        key = f"{source.id}:{source.collection.revision}:{strategy}:{target}:{sequence}:{first}:{last}"
        stable_id = hashlib.sha256(key.encode()).hexdigest()[:20]
        return Chunk(
            stable_id, source, tuple(title for _, title in headings),
            "\n\n".join(item.markdown for item in items), first, last,
        )

    for block in blocks:
        if block.heading_level and strategy == "heading":
            if staged:
                yield chunk(staged)
                staged, length = [], 0
            headings = [(level, title) for level, title in headings if level < block.heading_level]
            if block.text and block.text.casefold() != source.title.casefold():
                headings.append((block.heading_level, block.text))
            continue
        if block.heading_level:
            block = Block(f"## {block.text}", block.start, block.end)
        for part in bounded(block, target):
            size = token_count(part.markdown)
            if staged and length + size > target:
                yield chunk(staged)
                tail = staged[-1]
                tail_size = token_count(tail.text)
                can_overlap = (
                    strategy == "heading" and len(staged) > 1 and tail.language is None
                    and tail_size <= 50 and tail_size + size <= target
                )
                staged, length = ([tail], tail_size) if can_overlap else ([], 0)
            staged.append(part)
            length += size
    if staged:
        yield chunk(staged)
