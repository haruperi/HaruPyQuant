from selenium import webdriver
from selenium.webdriver.firefox.service import Service as FirefoxService
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.firefox import GeckoDriverManager
from selenium.common.exceptions import TimeoutException
from time import sleep
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
from datetime import datetime, timedelta
import pytz

# Simulate pressing tab n times
def tab(n=1):
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.TAB * n)

# Simulate pressing enter n times
def enter(n=1):
    driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.ENTER * n)

# Determine This Week's dates from the 2nd week, for which date range is already defined
def get_next_week(week_range):
    date_range_match = re.search(r'([A-Za-z]+-\d+)\s*-\s*([A-Za-z]+-\d+)', week_range)
    if date_range_match:
        start_date_str, end_date_str = date_range_match.groups()
        start_date = datetime.strptime(end_date_str, '%b-%d')
        end_date = datetime.strptime(start_date_str, '%b-%d').replace(year=start_date.year)
        next_start_date = end_date + timedelta(days=1)
        next_end_date = next_start_date + timedelta(days=6)
        next_week_range = f'{next_end_date.strftime("%b-%d")} - {next_start_date.strftime("%b-%d")}'
        return next_week_range
    return ''

# Determine date range for any of the 2nd set of weeks
def get_prev_week(week_range):
    date_range_match = re.search(r'([A-Za-z]+-\d+)\s*-\s*([A-Za-z]+-\d+)', week_range)
    if date_range_match:
        start_date_str, end_date_str = date_range_match.groups()
        start_date = datetime.strptime(end_date_str, '%b-%d')
        end_date = datetime.strptime(start_date_str, '%b-%d').replace(year=start_date.year)
        prev_start_date = start_date - timedelta(days=7)
        prev_end_date = end_date - timedelta(days=7)
        prev_week_range = f'{prev_end_date.strftime("%b-%d")} - {prev_start_date.strftime("%b-%d")}'
        prev_week_range = prev_week_range.replace('-0', '-')
        return prev_week_range
    return ''


# Load all articles and export a metadata file for each week
def fetch_news_from_msn():
    now = datetime.now(pytz.timezone('US/Eastern'))
    current_date = now.strftime("%Y-%m-%d")

    driver.get(news_url)

    # Wait for "Show More" button to appear
    WebDriverWait(driver, 5).until(
        EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Show More')]"))
    )

    try:
        # Process 2 sets of 4 weekly buttons each
        weekly_buttons = None
        for weekly_set_idx in range(2):
            if weekly_set_idx == 0 and not weekly_buttons:
                weekly_buttons = driver.find_elements(By.XPATH, "//button[contains(@class, 'timeRangeBox-DS-EntryPoint1-1')]")
            if weekly_set_idx == 1 and weekly_button_idx == 0:#and len(weekly_buttons) == 8:
                weekly_buttons = weekly_buttons[4:]
            # For each weekly button within the current set of 4
            for weekly_button_idx in range(4):
                if weekly_set_idx == 1 and weekly_button_idx == 3:
                    # Skip the last week in the 2nd set of weeks since button is disabled
                    print('Done fetching news!')
                    continue

                if weekly_set_idx == 0 and weekly_button_idx == 0:
                    date_range_text = weekly_buttons[1].find_element(By.CLASS_NAME, 'timeRangeName-DS-EntryPoint1-1').text
                    date_range_text = get_next_week(date_range_text)
                elif weekly_set_idx == 1:
                    date_range_text = get_prev_week(date_range_text)
                else:
                    # Skip weekly button if "No News" for that week, otherwise get date range
                    if weekly_buttons[weekly_button_idx].get_attribute('disabled'):
                        print('Weekly button disabled. Moving on.')
                        continue
                    else:
                        date_range_text = weekly_buttons[weekly_button_idx].find_element(By.CLASS_NAME, 'timeRangeName-DS-EntryPoint1-1').text

                # Initialize a list to store news headlines and metadata
                headline_metadata = []
                
                # Click the 'Show More' buttons until all articles are showing
                while True:
                    try:
                        # Wait for "Show More" buttons to appear
                        WebDriverWait(driver, timeout).until(
                            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Show More')]"))
                        )
                        show_more_buttons = driver.find_elements(By.XPATH, "//div[contains(text(), 'Show More')]")
                        
                        # Click on the first "Show More" button in sequence
                        first_show_more = show_more_buttons[0]
                        driver.execute_script("arguments[0].scrollIntoView(true);", first_show_more)
                        driver.execute_script("window.scrollBy(0, -200);")
                        try:
                            first_show_more.click()
                        except:
                            try:
                                driver.execute_script("window.scrollBy(0, 100);")
                                first_show_more.click()
                            except:
                                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                                first_show_more.click()
                            
                    except TimeoutException:
                        html_content = driver.page_source
                        
                        # Parse with BeautifulSoup
                        soup = BeautifulSoup(html_content, 'html.parser')
                        river_sections = soup.find_all('div', class_='financeFeedsSectionFourColumns-DS-EntryPoint1-1')

                        # Process all three sections of news cards
                        for idx in range(3):                        
                            # Find all article cards within each sentiment section
                            river_section = river_sections[idx]
                            river_articles = river_section.find_all('div', attrs={'data-t': True})

                            sentiment = None
                            for article in river_articles:
                                # Skip ads
                                if article.find('a', title="Ad"):
                                    continue

                                # Extract sentiment if not already known
                                if not sentiment:
                                    # sentiment = article.find_parent('div', class_='financeFeedsSectionFourColumns-DS-EntryPoint1-1').find('h1', class_='lightTitle-DS-EntryPoint1-1').text
                                    sentiment = article.find_parent('div', class_='sectionView-DS-EntryPoint1-1')['data-nav-text'].split(' ')[0].capitalize()

                                # Extract headline
                                headline_elem = article.find('h3')

                                # Skip blank cards by checking for headline
                                if not headline_elem:
                                    continue
                                
                                # Extract headline & link
                                headline = headline_elem.text.strip()
                                link_elem = article.find('a', href=True)
                                link = link_elem['href']
                                
                                # Extract provider and time posted from attribution section
                                provider_span = article.find_all('span', class_=lambda value: value and value.startswith('provider_name-DS-card1'))
                                provider = provider_span[0].text.strip()
                                time_posted = provider_span[-1].text.strip()

                                # Append to metadata ensuring provider and time posted are included
                                headline_metadata.append({
                                    'headline': headline,
                                    'link': link,
                                    'provider': provider,
                                    'time_posted': time_posted,
                                    'sentiment': sentiment
                                })

                        # Convert to DataFrame and drop duplicates
                        df_metadata = pd.DataFrame(headline_metadata)
                        df_metadata.drop_duplicates(inplace=True)

                        # Save to CSV
                        output_file = f'{output_dir}/{date_range_text}_{current_date}.csv'
                        df_metadata.to_csv(output_file, index=False, encoding='utf-8-sig')
                        print(f"Data saved to {output_file}")
                        break
                
                if weekly_set_idx == 0 and weekly_button_idx != 3:
                    # Click the next weekly button if it's in the first set
                    next_weekly_button = weekly_buttons[weekly_button_idx + 1]
                    driver.execute_script("arguments[0].scrollIntoView(true);", next_weekly_button)
                    driver.execute_script("window.scrollBy(0, -200);")
                    next_weekly_button.click()

                elif weekly_set_idx == 0 and weekly_button_idx == 3:
                    # Access the next set of weekly buttons and click the first one
                    next_button = driver.find_element(By.CLASS_NAME, "slideShowNextButton-DS-EntryPoint1-1")
                    driver.execute_script("window.scrollTo(0, 0);")
                    sleep(1)
                    next_button.click()
                    sleep(1)
                    sentiment_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Sentiment')]")
                    sentiment_button.click()
                    tab(3)
                    enter()
                    break

                elif weekly_set_idx == 1:
                    # Click the next weekly button if it's in the second set
                    sentiment_button = driver.find_element(By.XPATH, "//span[contains(text(), 'Sentiment')]")
                    sentiment_button.click()
                    tab(4 + weekly_button_idx)
                    enter()

    finally:
        driver.quit()

news_url = 'https://www.msn.com/en-us/money/markets?l3=Index_L3_Sentiment'
output_dir = 'msn-news'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

firefox_options = FirefoxOptions()
firefox_options.add_argument("--headless")

timeout = 2  # Max seconds to wait for new articles to load
driver = webdriver.Firefox(service=FirefoxService(GeckoDriverManager().install()), options=firefox_options)
driver.set_window_size(900, 700)

fetch_news_from_msn()