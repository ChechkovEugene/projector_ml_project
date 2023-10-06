import os
import pandas as pd
import requests

from bs4 import NavigableString, Comment
from bs4 import BeautifulSoup

PARSED_DATA_FOLDER = "parsed_data"
URL = "https://winetime.com.ua/ua/wine"

COLUMN_NAMES = ['Назва','Виробник','Артикул','Ціна','Температура подачі','Сорти винограду',
                'Технологія виробництва','Об `єм','Рік','Бренд','Регіон','Країна',
                'Солодкість','Тип напою', 'Колір вина','Склад землі','З чим подавати','Класифікація',
                'Розміщення виноградників','Витримка','Склад винограду','Цукор',
                'Алкоголь, %','Збір урожаю','Розширений колір вина','Аромат','Смак','Цікаве',
                'Стиль вин','Потенціал','Дегустації']

def main() -> None:
    wines_data = []

    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'html.parser')
    total_count = soup.select_one('span.total-catalog-items').text
    total_pages = round(int(total_count)/30)

    for page in range(1, total_pages + 1):
        print(f'Page processing: {page} of {total_pages}')
        new_page_url = URL + f"?page={page}"
        page = requests.get(new_page_url)
        soup = BeautifulSoup(page.content, 'html.parser')
        products = soup.select_one('div.catalog-list-wrapper').select('div.products-main-slider-item')

        for i in range(0, len(products)):
            product = products[i]
            product_url = product.select_one('a')['href'] + "#description"
            item_page = requests.get(product_url)
            item_soup = BeautifulSoup(item_page.content, 'html.parser')

            description_soup = item_soup.select_one("div.description-tab")
            title = description_soup.select_one('div.item-list-title')
            title_text = title.text.strip() if title else ''

            print(f"{title_text}: {product_url}")

            subtitle = description_soup.select_one('div.item-list-subtitle')
            subtitle_text = subtitle.text.strip() if subtitle else ''

            vendor_code = description_soup.select_one('div.vendor-code').select_one('span')
            vendor_code_text = vendor_code.text.strip() if vendor_code else ''

            price = description_soup.select_one('div.own-bottom').select_one('span')
            price_text = price.text.strip() if price else ''

            tables = item_soup.select('table.char-item-table')
            wine = dict.fromkeys(COLUMN_NAMES)
            wine['Назва'] = title_text
            wine['Виробник'] = subtitle_text
            wine['Артикул'] = vendor_code
            wine['Ціна'] = price_text

            for table in tables:
                rows = table.select('tr')
                for row in rows:
                    row_name = row.select_one('td.first-char-title').text.strip()
                    row_value = row.select_one('td.second-char-title').text.strip()
                    wine[row_name] = row_value

            wines_data.append(wine)
        df = pd.DataFrame(wines_data)
        df.to_csv(os.path.join(PARSED_DATA_FOLDER, "wines.csv"), mode='w+')


if __name__ == "__main__":
    main()
