import requests

class Developer:

    salary = 100_000

    def __init__(self, first_name, last_name, fav_language):
        self.first_name = first_name
        self.last_name = last_name
        self.fav_language = fav_language

    @property
    def email(self):
        return f'{self.first_name}.{self.last_name}@gmail.com'

    @property
    def fullname(self):
        return '{} {}'.format(self.first_name, self.last_name)

    def give_raise(self):
        return self.salary + 10_000

    def scrape_webpage(self):
        response = requests.get("https://outlook.live.com/mail/0/")
        if response.ok:
            return response.text
        else:
            return 'Bad Response'
