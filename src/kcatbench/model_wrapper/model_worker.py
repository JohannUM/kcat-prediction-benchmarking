import argparse
import pandas as pd
from pathlib import Path
from collections.abc import Callable

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

PREDICT_HANDLERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "dlkcat": _predict_dlkcat,
    "catapro": _predict_catapro,
    "catpred": _predict_catpred
}

def run_predict(model:str, input_path:Path, output_path:Path):
    if model not in PREDICT_HANDLERS:
        raise ValueError(f"Unknown model '{model}'.")
    
    input = pd.read_csv(input_path, sep="\t")
    output = PREDICT_HANDLERS[model](input)
    output.to_csv(output_path, sep="\t", index=False)

def main(argv: list[str] | None = None) -> None:
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