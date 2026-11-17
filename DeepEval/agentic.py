from deepeval.dataset import Golden, EvaluationDataset
from deepeval.metrics import TaskCompletionMetric
from deepeval.tracing import observe

@observe()
def trip_planner_agent(input):
    destination = "Paris"
    days = 2

    @observe()
    def restaurant_finder(city):
        return ["Le Jules Verne", "Angelina Paris", "Septime"]

    @observe()
    def itinerary_generator(destination, days):
        return ["Eiffel Tower", "Louvre Museum", "Montmartre"][:days]

    itinerary = itinerary_generator(destination, days)
    restaurants = restaurant_finder(destination)

    return itinerary + restaurants

dataset = EvaluationDataset(goldens=[Golden(input="This is a test query")])

task_completion = TaskCompletionMetric(threshold=0.7, model="gpt-4o")

for golden in dataset.evals_iterator(metrics=[task_completion]):
    trip_planner_agent(golden.input)