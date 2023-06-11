import os
import requests
from bs4 import BeautifulSoup
from dataclasses import dataclass, field
from tqdm import tqdm

from dataclasses import dataclass
from pathlib import Path

FOLDER_PATH = "dataset/new"
WEBSITE = "https://pl.wikipedia.org/wiki/"
HEADER = {
    "User-Agent": "CoolBot/0.0 (https://example.org/coolbot/; coolbot@example.org)"
}


@dataclass
class WikipediaImageScraper:
    source_dir: Path = Path("dataset/new")
    wikipedia_url: str = "https://pl.wikipedia.org/wiki/"
    _data: dict = field(init=False, default_factory=dict)

    def __post_init__(self):
        os.makedirs(self.source_dir, exist_ok=True)

    def _download_image(self, name: str, url: str):
        ext = str(url[-3:]).lower()
        response = requests.get(url, headers=HEADER)

        if response.ok:
            file_path = self.source_dir / name / f"img2.{ext}"
            with open(file_path, "wb") as f:
                f.write(response.content)

    def _get_image_url(self, url: str) -> str:
        response = requests.get(url)
        html_content = response.text

        soup = BeautifulSoup(html_content, "html.parser")

        for alt in ["Ilustracja", "ilustracja"]:
            image = soup.find("img", alt=alt)
            if image is not None:
                image_path = "https:" + image["src"]
                return image_path
            return None

    def run(self):
        for name in tqdm(os.listdir(self.source_dir)):
            item = "_".join(name.split(" "))
            target_url = self.wikipedia_url + item
            image_url = self._get_image_url(url=target_url)

            if image_url:
                self._download_image(name, image_url)


if __name__ == "__main__":
    scraper = WikipediaImageScraper()
    scraper.run()
