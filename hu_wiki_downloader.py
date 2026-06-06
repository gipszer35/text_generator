from datasets import load_dataset
import os
import re

ds = load_dataset(
    "wikimedia/wikipedia",
    "20231101.hu",
    split="train",
    streaming=True
)

output_dir = "wikipedia_hu_wiki"
os.makedirs(output_dir, exist_ok=True)

BAD_TITLES = {
    "Diszkográfia",
    "Fordítás",
    "Források",
    "Irodalom",
    "Jegyzetek",
    "Kezdőlapon szerepelt szócikkek",
    "Kitalált személyek",
    "Külföldiek",
    "Magyarok",
    "Magyarul",
    "Megjegyzések",
    "Munkái",
    "Munkája",
    "Művei",
    "Névnapok",
    "További információk"
}

year_pattern = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")


def has_too_many_numbers(line):
    numbers = re.findall(r"\b\d+\b", line)
    return len(numbers) > 3


def year_count(line):
    return len(year_pattern.findall(line))


def is_bad_line(line):
    raw = line.strip()

    # Remove short lines
    if len(raw) < 80:
        return True

    # starts with a year/date
    m = re.match(r"^\s*(\d{1,4})\b", raw)
    if m:
        num = int(m.group(1))
        if 0 <= num <= 2999:
            return True

# remove lines starting with a month (timeline entries)
    if re.match(
        r"^\s*(január|február|március|április|május|június|július|augusztus|szeptember|október|november|december)\b",
        raw,
        re.IGNORECASE
    ):
        return True

    # More than 3 numeric values
    if has_too_many_numbers(raw):
        return True

    yc = year_count(raw)

    # More than 4 year references
    if yc > 4:
        return True

    # Short line containing a year
    if len(raw) < 100 and yc >= 1:
        return True

    return False


def safe_filename(text):
    text = text or "article"
    text = re.sub(r"[^a-zA-Z0-9áéíóöőüűÁÉÍÓÖŐÜŰ]+", "_", text)
    return text[:120].strip("_")


for i, article in enumerate(ds):

    text = article["text"]

    if not text:
        continue

    lines = text.split("\n")
    kept = []

    for line in lines:
        raw = line.strip()  # remove leading/trailing whitespace

        # remove empty lines
        if raw == "":
            continue

        # stop when appendix/reference section starts
        if raw in BAD_TITLES:
            break

        if not is_bad_line(raw):
            kept.append(raw)

    # Count only substantial lines
    content_lines = len(kept)

    if content_lines < 10:
        continue

    final_text = "\n".join(kept)

    title = article.get("title", f"article_{i}")
    filename = f"{i:06d}_{safe_filename(title)}.wiki"
    path = os.path.join(output_dir, filename)

    with open(path, "w", encoding="utf-8") as f:
        f.write(final_text)

    if i % 1000 == 15:
        print("processed:", i)
