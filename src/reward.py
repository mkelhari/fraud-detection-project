def calculate_reward(action, real_class):
    """
    action:
        0 = accepter
        1 = bloquer

    real_class:
        0 = transaction normale
        1 = fraude
    """

    # Transaction normale acceptée
    if action == 0 and real_class == 0:
        return 1

    # Fraude bloquée
    if action == 1 and real_class == 1:
        return 5

    # Fraude acceptée = erreur grave
    if action == 0 and real_class == 1:
        return -10

    # Transaction normale bloquée
    if action == 1 and real_class == 0:
        return -2