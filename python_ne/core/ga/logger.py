import time
import csv


class GaLogger:

    def __init__(self, console_log=True):
        self.logged_data = []
        self.generation_start_time = 0
        self.console_log = console_log

    def start_generation_log(self):
        self.generation_start_time = time.time()

    def finish_log_generation(self, generation, best_element_fitness):
        time_to_run_generation = time.time() - self.generation_start_time
        data = {
            'generation': generation,
            'best_element_fitness': best_element_fitness,
            'time_to_run_generation': time_to_run_generation
        }
        self.logged_data.append(data)

        if self.console_log:
            print(data)

    def save_as_csv(self, file_path):
        file = open(file_path, 'w+')
        csv_writer = csv.DictWriter(file, fieldnames=self.logged_data[0].keys())

        csv_writer.writeheader()

        for data in self.logged_data:
            csv_writer.writerow(data)

        file.close()
