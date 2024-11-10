"""
Cumulate quantities, keep counts, and retrun averages
"""
from collections import defaultdict

class Cumulator:
    """
    Cumulate quantities, keep counts, and retrun averages
    """
    def __init__(self):
        self.total = defaultdict(float)
        self.count = defaultdict(int)

    def update(self, new_data):
        for key, val in new_data.items():
            self.count[key] += 1
            self.total[key] += val

    def get_average(self, decimal_places=6):
        template = f'{{num:.{decimal_places}f}}'
        return {key: template.format(num=self.total[key] / self.count[key])
                for key in self.total}
