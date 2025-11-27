import json
from pathlib import Path

import structlog
import tqdm
import typer
from tencent_cloud_hackathon_intelligent_pentest_competition_api_server.models.benchmark import Benchmark


app = typer.Typer()
logger = structlog.get_logger()


@app.command()
def main(
    xbow_benchmark_folder: Path = typer.Option(
        Path('~/xbow-validation-benchmarks').expanduser(),
        help='Path to the Xbow benchmark folder (e.g., ~/xbow-validation-benchmarks)',
    ),
    output_file: Path = typer.Option(
        Path('benchmarks.json'),
        help='Output path for the processed JSON file',
    ),
):
    logger.info(
        'collecting benchmark metadata',
        xbow_benchmark_folder=xbow_benchmark_folder, output_file=output_file,
    )
    input_folder = xbow_benchmark_folder.expanduser()
    if not input_folder.exists():
        logger.error('folder does not exist', folder=input_folder)
        raise typer.Exit(code=1)

    benchmarks_data = []
    paths = list(input_folder.glob('benchmarks/*/*.json'))

    for path in tqdm.tqdm(paths, desc='processing benchmarks'):
        with open(path, encoding='utf-8') as f:
            raw_data = json.load(f)
            benchmark_id = path.parent.stem
            raw_data['id'] = benchmark_id
            model = Benchmark.model_validate(raw_data)
            benchmarks_data.append(model.model_dump(mode='json'))
            logger.debug(
                'benchmark processed',
                benchmark_id=benchmark_id, benchmark=model,
            )

    benchmarks_data.sort(key=lambda x: x['id'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(
            benchmarks_data, f, indent=4,
            ensure_ascii=False, sort_keys=True,
        )

    logger.info(
        'benchmark metadata saved', output_file=output_file,
        num_benchmarks=len(benchmarks_data),
    )


if __name__ == '__main__':
    app()
