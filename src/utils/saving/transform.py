import pickle


def save_custom_transform_object(obj, path: str, name: str, type: str):
    with open() as f:
        pickle.dump(obj, f)


# def save_custom_transform_struct(class_, path: str, name: str, type: str):
#     ...


def save_torch_transform_object(class_, path: str, name: str, type: str):
    ...