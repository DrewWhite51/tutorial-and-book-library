
class Developer:

    salary = 100_000

    def __init__(self, first_name, last_name, fav_language):
        self.first_name = first_name,
        self.last_name = last_name
        self.fav_language = fav_language

    @property
    def email(self):
        return '{}.{}@gmail.com'.format(self.first_name, self.last_name)

    @property
    def fullname(self):
        return '{} {}'.format(self.first_name, self.last_name)

    def give_raise(self):
        self.salary = self.salary + 10_000
