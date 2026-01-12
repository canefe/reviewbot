from pydantic import RootModel


class ReviewSummary(RootModel[str]):
    def __str__(self) -> str:
        return self.root

    def __len__(self) -> int:
        return len(self.root)
