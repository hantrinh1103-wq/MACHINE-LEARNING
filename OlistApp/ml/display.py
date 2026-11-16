"""Persistent display IDs; original keys remain unchanged in source data."""
import csv
import os
import tempfile
import unicodedata
from pathlib import Path


STATE_NAMES = {
    "AC": "Acre", "AL": "Alagoas", "AP": "Amapá", "AM": "Amazonas",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal",
    "ES": "Espírito Santo", "GO": "Goiás", "MA": "Maranhão",
    "MT": "Mato Grosso", "MS": "Mato Grosso do Sul", "MG": "Minas Gerais",
    "PA": "Pará", "PB": "Paraíba", "PR": "Paraná", "PE": "Pernambuco",
    "PI": "Piauí", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RS": "Rio Grande do Sul", "RO": "Rondônia", "RR": "Roraima",
    "SC": "Santa Catarina", "SP": "São Paulo", "SE": "Sergipe", "TO": "Tocantins",
}


def state_label(value):
    code = str(value).strip().upper()
    if code in ("", "NAN", "NONE"):
        return "Chưa có thông tin"
    name = STATE_NAMES.get(code)
    return f"{name} ({code})" if name else code


def search_text(value):
    """Allow searches such as 'sao paulo' as well as 'São Paulo'."""
    return "".join(c for c in unicodedata.normalize("NFKD", str(value).casefold())
                   if not unicodedata.combining(c))


class DisplayIds:
    def __init__(self, path, originals, prefix, width):
        self.path = Path(path)
        self.by_original = {}
        self.by_display = {}
        next_number = 1
        if self.path.exists():
            with self.path.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    original, display = row["original_id"], row["display_id"]
                    suffix = display.removeprefix(prefix + "-")
                    if (not original or original in self.by_original or display in self.by_display
                            or not display.startswith(prefix + "-") or not suffix.isdigit()
                            or int(suffix) < 1):
                        raise ValueError(f"Bảng mã hiển thị không hợp lệ: {self.path}")
                    self.by_original[original] = display
                    self.by_display[display] = original
                    next_number = max(next_number, int(suffix) + 1)
        missing = sorted(set(map(str, originals)) - self.by_original.keys())
        for original in missing:
            display = f"{prefix}-{next_number:0{width}d}"
            self.by_original[original] = display
            self.by_display[display] = original
            next_number += 1
        if missing:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            # Replace only after a complete write, so an interrupted save keeps the old map.
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                                 dir=self.path.parent, delete=False) as stream:
                    temporary = Path(stream.name)
                    writer = csv.writer(stream)
                    writer.writerow(["original_id", "display_id"])
                    writer.writerows(self.by_original.items())
                os.replace(temporary, self.path)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)

    def display(self, original):
        return self.by_original.get(str(original), str(original))

    def resolve(self, value):
        value = str(value).strip()
        return self.by_display.get(value.upper(), value.lower())
