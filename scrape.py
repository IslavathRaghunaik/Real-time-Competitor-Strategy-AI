# import json
# import time
# from datetime import datetime

# import pandas as pd
# from selenium import webdriver
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.common.by import By
# from selenium.webdriver.support import expected_conditions as EC
# from selenium.webdriver.support.wait import WebDriverWait
# from webdriver_manager.chrome import ChromeDriverManager

# links = {
#     "Apple iPhone 15": "https://www.amazon.in/dp/B0CHX3TW6X?ref=ods_ucc_kindle_B0CHX2WQLX&th=1",
#     "Apple 2023 MacBook Pro (16-inch, Apple M3 Pro chip with 12‑core CPU and 18‑core GPU, 36GB Unified Memory, 512GB) - Silver": "https://amzn.in/d/ib419CQ",
#     "OnePlus Nord 4 5G (Mercurial Silver, 8GB RAM, 256GB Storage)": "https://amzn.in/d/2KOJBxa",
#     "Sony WH-1000XM5 Best Active Noise Cancelling Wireless Bluetooth Over Ear Headphones with Mic for Clear Calling, up to 40 Hours Battery -Black": "https://amzn.in/d/dP5ATPJ",
# }


# def scrape_product_data(link):
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--lang=en")
#     options.add_argument("--window-size=1920,1080")

#     driver = webdriver.Chrome(
#         service=Service(ChromeDriverManager().install()), options=options
#     )
#     driver.set_window_size(1920, 1080)
#     driver.get(link)
#     product_data, review_data = {}, {}
#     product_data["reviews"] = []
#     wait = WebDriverWait(driver, 10)
#     time.sleep(5)
#     retry = 0
#     while retry < 3:
#         try:
#             driver.save_screenshot("screenshot.png")
#             wait.until(EC.presence_of_element_located((By.CLASS_NAME, "a-offscreen")))
#             break
#         except Exception:
#             print("retrying")
#             retry += 1
#             driver.get(link)
#             time.sleep(5)

#     driver.save_screenshot("screenshot.png")
#     try:
#         price_elem = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[3]/span[2]/span[2]',
#         )
#         product_data["selling price"] = int("".join(price_elem.text.strip().split(",")))
#     except:
#         product_data["selling price"] = 0

#     try:
#         original_price = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[2]/span/span[1]/span[2]/span/span[2]',
#         ).text
#         product_data["original price"] = int("".join(original_price.strip().split(",")))
#     except:
#         product_data["original price"] = 0

#     try:
#         discount = driver.find_element(
#             By.XPATH,
#             '//*[@id="corePriceDisplay_desktop_feature_div"]/div[1]/span[2]',
#         )
#         full_rating_text = discount.get_attribute("innerHTML").strip()
#         if " out of 5 stars" in full_rating_text.lower():
#             product_data["rating"] = (
#                 full_rating_text.lower().split(" out of")[0].strip()
#             )
#         else:
#             product_data["discount"] = full_rating_text
#     except:
#         product_data["discount"] = 0

#     try:
#         driver.find_element(By.CLASS_NAME, "a-icon-popover").click()
#         time.sleep(1)
#     except:
#         pass

#     try:
#         reviews_link = driver.find_elements(
#             By.XPATH, "//a[contains(text(),'See customer reviews')]"
#         )[-1].get_attribute("href")
#         product_data["product_url"] = reviews_link.split("#")[0]
#         driver.get(reviews_link)
#         time.sleep(3)
#         reviews = driver.find_element(By.ID, "cm-cr-dp-review-list")
#         reviews = reviews.find_elements(By.TAG_NAME, "li")
#         for item in reviews:
#             product_data["reviews"].append(item.get_attribute("innerText"))

#         driver.back()
#         time.sleep(3)
#     except Exception:
#         product_data["reviews"] = []

#     product_data["date"] = time.strftime("%Y-%m-%d")
#     review_data["date"] = time.strftime("%Y-%m-%d")
#     driver.quit()
#     return product_data


# for product_name, link in links.items():
#     product_data = scrape_product_data(link)
#     reviews = json.loads(pd.read_csv("reviews.csv").to_json(orient="records"))
#     price = json.loads(pd.read_csv("competitor_data.csv").to_json(orient="records"))
#     price.append(
#         {
#             "product_name": product_name,
#             "Price": product_data["selling price"],
#             "Discount": product_data["discount"],
#             "Date": datetime.now().strftime("%Y-%m-%d"),
#         }
#     )
#     for i in product_data["reviews"]:
#         reviews.append({"product_name": product_name, "reviews": i})

#     pd.DataFrame(reviews).to_csv("reviews.csv", index=False)
#     pd.DataFrame(price).to_csv("competitor_data.csv", index=False)

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import time
import random

def scrape_product_data(url):
    """Scrape product data from e-commerce website."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extract product details
        product_name = soup.find('span', {'class': 'product-title'}).text.strip()
        price = float(soup.find('span', {'class': 'price'}).text.replace('₹', '').replace(',', '').strip())
        discount = float(soup.find('span', {'class': 'discount'}).text.replace('%', '').strip())
        
        # Get reviews if available
        reviews = []
        review_elements = soup.find_all('div', {'class': 'review'})
        for review in review_elements[:5]:  # Get latest 5 reviews
            review_text = review.find('p').text.strip()
            rating = float(review.find('span', {'class': 'rating'}).text.strip())
            reviews.append({
                'review_text': review_text,
                'rating': rating,
                'date': datetime.now().strftime('%Y-%m-%d')
            })
        
        return {
            'product_data': {
                'product_name': product_name,
                'price': price,
                'discount': discount,
                'date': datetime.now().strftime('%Y-%m-%d')
            },
            'reviews': reviews
        }
        
    except Exception as e:
        print(f"Error scraping data: {str(e)}")
        return None

def update_csv_files(product_data, reviews):
    """Update CSV files with new data."""
    # Update competitor_data.csv
    try:
        df_products = pd.DataFrame([product_data])
        df_products.to_csv('competitor_data.csv', mode='a', header=False, index=False)
        
        # Update reviews.csv
        if reviews:
            df_reviews = pd.DataFrame(reviews)
            df_reviews.to_csv('reviews.csv', mode='a', header=False, index=False)
            
        return True
    except Exception as e:
        print(f"Error updating CSV files: {str(e)}")
        return False

def main():
    # List of competitor URLs to monitor
    competitor_urls = [
        'https://example.com/product1',
        'https://example.com/product2',
        # Add more URLs as needed
    ]
    
    while True:
        for url in competitor_urls:
            data = scrape_product_data(url)
            if data:
                update_csv_files(data['product_data'], data['reviews'])
            
            # Random delay between requests (2-5 seconds)
            time.sleep(random.uniform(2, 5))
        
        # Wait for 1 hour before next update
        time.sleep(3600)

if __name__ == "__main__":
    main()