from bs4 import BeautifulSoup
import requests
url = "https://www.gutenberg.org/cache/epub/8800/pg8800-images.html"

response = requests.get(url)
data = response.text

# Parse the HTML content
soup = BeautifulSoup(data, 'html.parser')
chapters = soup.find_all('div', attrs = {'class':'chapter'})

for chapter in chapters :
    # chapter title
    title = chapter.find('h2').text
    # chapter content
    content = ' '.join ([itr.text for itr in chapter.find_all('p')])
    content = ' '.join(content.split("\r\n"))
    f = open("data2.txt", "a")
    f.write(title + "\t" + content.replace("\n", " ").strip() + "\n")
    f.close()
