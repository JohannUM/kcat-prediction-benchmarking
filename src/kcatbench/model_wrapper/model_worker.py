import argparse
import sys
import traceback
import pandas as pd
from pathlib import Path
from collections.abc import Callable
from typing import Optional

from kcatbench.util import read_csv_with_schema

def _predict_dlkcat(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.dlkcat_wrapper import DLKcatWrapper
    model = DLKcatWrapper()
    return model.predict(input)

def _predict_catapro(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.catapro_wrapper import CataProWrapper
    model = CataProWrapper()
    return model.predict(input)

def _predict_catpred(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.catpred_wrapper import CatPredWrapper
    model = CatPredWrapper()
    return model.predict(input)

def _predict_turnup(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.turnup_wrapper import TurNuPWrapper
    model = TurNuPWrapper()
    return model.predict(input)

def _predict_unikp(input:pd.DataFrame) -> pd.DataFrame:
    from kcatbench.model_wrapper.inner_wrapper.unikp_wrapper import UniKPWrapper
    model = UniKPWrapper()
    return model.predict(input)

PREDICT_HANDLERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "dlkcat": _predict_dlkcat,
    "catapro": _predict_catapro,
    "catpred": _predict_catpred,
    "turnup": _predict_turnup,
    "unikp": _predict_unikp
}

def run_predict(model:str, input_path:Path, output_path:Path):
    if model not in PREDICT_HANDLERS:
        raise ValueError(f"Unknown model '{model}'.")
    
    try:
        input = read_csv_with_schema(input_path)

        output = PREDICT_HANDLERS[model](input)

        output.to_csv(output_path, index=False)

    except Exception as e:
        print(f"--- WORKER EXCEPTION IN MODEL: {model} ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Conda worker that runs a kcat prediction model")
    
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--model", type=str, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)

    # Maybe in the future
    # train_parser = subparsers.add_parser("train")

    args = parser.parse_args(argv)

    if args.command == "predict":
        run_predict(args.model, args.input, args.output)
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()