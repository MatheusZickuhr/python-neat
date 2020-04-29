import time
import csv

from python_ne.utils.observer import Observer


class GaLogger(Observer):

    def notify(self, *args, **kwargs):

        data = {
            'generation': kwargs['current_generation'],
            'best_element_fitness': kwargs['best_element_fitness'],
            'time_to_run_generation': kwargs['generation_time']
        }

        self.logged_data.append(data)

        print(f'generation={kwargs["current_generation"]}/{kwargs["number_of_generations"]},' +
              f' bestfitness={kwargs["best_element_fitness"]}, runtime={kwargs["generation_time"]}')

    def __init__(self):
        self.logged_data = []
        self.generation_start_time = 0

    def save_as_csv(self, file_path):
        file = open(file_path, 'w+')
        csv_writer = csv.DictWriter(file, fieldnames=self.logged_data[0].keys())

        csv_writer.writeheader()

        for data in self.logged_data:
            csv_writer.writerow(data)

        file.close()
