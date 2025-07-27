import numpy as np
from pathlib import Path
from typing import Literal, List, Union
import argparse

class EnsembleUnifier:
    def __init__(self, root_dir: Union[str, Path], method: Literal["mean", "median"] = "mean"):
        self.root_dir = Path(root_dir)
        self.method = method
        self.runs = self._discover_runs()
        self.available_dates = self._discover_dates()

        self.unified_dir = self.root_dir / "unified" / self.method
        self.unified_dir.mkdir(parents=True, exist_ok=True)

    def _discover_runs(self) -> List[Path]:
        return sorted([
            p / "files" / "predictions"
            for p in self.root_dir.iterdir()
            if p.is_dir() and p.name != "unified"
        ])

    def _discover_dates(self) -> List[str]:
        date_sets = []
        for run in self.runs:
            files = run.glob("rea_data_*.npy")
            dates = {f.name.split("_")[-1].replace(".npy", "") for f in files}
            date_sets.append(dates)
        return sorted(set.intersection(*date_sets)) if date_sets else []

    def unify_date(self, date_str: str) -> np.ndarray:
        arrays = []
        for run in self.runs:
            file_path = run / f"rea_data_{date_str}.npy"
            arr = np.load(file_path)
            arrays.append(arr)

        stacked = np.stack(arrays, axis=0)
        
        if self.method == "mean":
            return np.nanmean(stacked, axis=0)
        elif self.method == "median":
            return np.nanmedian(stacked, axis=0)
        else:
            raise ValueError(f"Método no válido: {self.method}")

    def unify_all(self, overwrite: bool = False):
        for date_str in self.available_dates:
            out_file = self.unified_dir / f"rea_data_{date_str}.npy"
            
            if not overwrite and out_file.exists():
                print(f"Saltando {date_str} (ya existe)")
                continue

            print(f"Unificando y guardando {date_str}...")
            result = self.unify_date(date_str)
            np.save(out_file, result)

def main():
    parser = argparse.ArgumentParser(description="Unifica predicciones de un conjunto de modelos.")
    parser.add_argument("root_dir", type=str, help="Directorio raíz que contiene las ejecuciones del conjunto.")
    parser.add_argument("--method", choices=["mean"], default="mean", help="Método de unificación (mean o median).")
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribir archivos ya existentes.")

    args = parser.parse_args()

    unifier = EnsembleUnifier(root_dir=args.root_dir, method=args.method)
    unifier.unify_all(overwrite=args.overwrite)

if __name__ == "__main__":
    main()