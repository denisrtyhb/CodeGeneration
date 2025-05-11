from base_utils import BaseClass, add_numbers, multiply_by_two


def process_data(data):
    """Processes data using functions from base_utils."""
    doubled = multiply_by_two(data)
    result = add_numbers(doubled, 5)  # Use the base_utils.add_numbers function
    return result


def create_and_greet(name):
    """Creates an instance of BaseClass and returns its greeting."""
    obj = BaseClass(name)  # Use the BaseClass from base_utils
    return obj.greet()

def main():
    # Example usage
    data = 10
    processed_value = process_data(data)
    print(f"Processed value: {processed_value}")

    greeting = create_and_greet("MyObject")
    print(greeting)

if __name__ == "__main__":
    main()