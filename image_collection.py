from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

from PIL import Image
from time import sleep
import requests
import os

# todays rankings
ai_page = "https://www.pixiv.net/ranking.php?mode=daily_ai"

path = os.getcwd()
usr_data_path = r"C:\Users\jiaho\AppData\Local\Google\Chrome\User Data\default"
script_url = r"https://greasyfork.org/scripts/369815-i-pximg-net-403-forbidden-fix/code/ipximgnet%20403%20Forbidden%20Fix.user.js"
button_xpath = "//*[@id=\"input_SW5zdGFsbF91bmRlZmluZWQ_bu\"]"

request_url = 'https://i.pximg.net/img-original/img/2023/07/18/03/23/09/110015970_p0.png'


url = 'https://www.pixiv.net/en/artworks/109807312'

# sets up chrome env
options = webdriver.ChromeOptions()
options.add_argument("--no-sandbox")
options.add_argument("--enable-sync")
#options.add_extension(r"C:\Users\micha\AppData\Local\Google\Chrome\User Data\Default\Extensions\dhdgffkkebhmkfjojejmpbldmpobfkfo\4.19.0_0")
options.add_argument(r'--load-extension=C:\Users\micha\AppData\Local\Google\Chrome\User Data\Default\Extensions\dhdgffkkebhmkfjojejmpbldmpobfkfo\4.19.0_0')

s = Service('./chromedriver.exe')
driver = webdriver.Chrome(service=s, options=options)

# installs extensions and vv slowly gets back to the original tab
sleep(3)
driver.get(script_url)
print('opened script url')
sleep(5)

windows = driver.window_handles
for w in windows:
  if 'chrome-extension' in driver.title: break
  print(w)
  driver.switch_to.window(w)
  sleep(3)

sleep(3)
driver.find_element(By.XPATH,button_xpath).click()
print('extension installed')
sleep(3)

windows = driver.window_handles
for w in windows:
  print(w)
  driver.switch_to.window(w)
  if '' in driver.title: break
  sleep(3)


from datetime import datetime,timedelta

# scrapes all the master img links from the rankings for that date (50 img each)
def getLinksFromDate(date,ai=True):

  formatted_date = str(date).split(' ')[0].replace('-','')
  if ai: header = 'https://www.pixiv.net/ranking.php?mode=daily_ai&date=' 
  else: header = 'https://www.pixiv.net/ranking.php?mode=daily&content=illust&date='
  page = header + formatted_date
  driver.get(page)

  sleep(5)
  
  img_links = []
  items = driver.find_elements(By.CLASS_NAME,"_layout-thumbnail")
  for item in items:
    item = item.get_attribute('outerHTML')
    for s in item.split(' '):
      if 'data-src' in s:
        img_links.append(s.replace('data-src=','').replace(r'"',''))
        break
  return img_links

all_ai_links = []
days_ago = 1

num_imgs = 5000

# gets unique ai img links
while len(all_ai_links) < num_imgs:
  date = datetime.now()-timedelta(days_ago)
  all_ai_links += getLinksFromDate(date,ai=True)
  all_ai_links = [*{*all_ai_links}]
  days_ago += 1

all_artist_links = []
days_ago = 1

# gets 200+ unique artist img links
while len(all_artist_links) < num_imgs:
  date = datetime.now()-timedelta(days_ago)
  all_artist_links += getLinksFromDate(date,ai=False)
  all_artist_links = [*{*all_artist_links}]
  days_ago += 1

print('finished reading all links')

img_xpath = '/html/body/img'


# added new code here may or may not work .-.
filenames = glob.glob('ai_images/*.png')+glob.glob('artist_images/*.png')
filenames = {s.replace('artist_images\\','') for s in filenames}

for i,link in enumerate(all_ai_links):
  img_title = link.split('/')[-1].replace('.jpg','.png')
  if img_title in filenames: continue
  driver.get(link)
  for i in range(5):
    sleep(3)
    try:
      img = driver.find_element(By.XPATH,img_xpath)
      sleep(1)
      img.screenshot(f'ai_images/{img_title}')
      break
    except:
      print(f'retrying ({i})')

for link in all_artist_links:
  img_title = link.split('/')[-1].replace('.jpg','.png')
  if img_title in filenames: continue
  driver.get(link)
  for i in range(5):
    sleep(3)
    try:
      img = driver.find_element(By.XPATH,img_xpath)
      sleep(1)
      img.screenshot(f'artist_images/{img_title}')
      break
    except:
      print(f'retrying ({i})')
  
  



    
      


