import json
from enum import Enum
from pathlib import Path

import typer
from pydantic import BaseModel
from pydantic import computed_field
from pydantic import Field

app = typer.Typer()


class WinCondition(str, Enum):
    FLAG = 'flag'


class Benchmark(BaseModel):
    id: str = Field(..., description='The id of the benchmark')
    name: str = Field(..., description='The name of the benchmark')
    description: str = Field(
        ...,
        description='The description of the benchmark',
    )
    level: int = Field(..., description='The level of the benchmark')
    win_condition: WinCondition = Field(
        ...,
        description='The win condition of the benchmark',
    )
    tags: list[str] = Field(..., description='The tags of the benchmark')
    canaries: list[str] = Field(
        ...,
        description='The canaries of the benchmark',
    )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def points(self) -> int:
        points = {
            1: 200,
            2: 300,
            3: 500,
        }
        return points[self.level]


def load_all_benchmarks(xbow_benchmark_folder: Path) -> list[Benchmark]:
    benchmarks = []
    for path in xbow_benchmark_folder.glob('benchmarks/*/*.json'):
        with open(path) as f:
            print(path)
            data = json.load(f)
            data['id'] = path.parent.stem
            benchmarks.append(Benchmark.model_validate(data))
    return benchmarks


@app.command()
def main(
    xbow_benchmark_folder: Path = typer.Option(
        Path('~/xbow-validation-benchmarks').expanduser(),
        help='Path to the Xbow benchmark folder (e.g., ~/xbow-validation-benchmarks)',
    ),
):
    benchmarks = load_all_benchmarks(xbow_benchmark_folder)
    print(benchmarks)


if __name__ == '__main__':
    app()
