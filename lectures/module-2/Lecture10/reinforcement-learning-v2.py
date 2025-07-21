import random 

# FOR 2 Dimensional 

#Environment Variables
area_size = 10
n_states = area_size * area_size # 100
goal_state =  n_states - 1 #99 parça için [9,9]
actions = ["up", "down", "left", "right"]

#Q Table
q_table = [[0 for _ in actions] for _ in range(n_states)] #HAFIZA, deneyim defteri

#Hyperparameters
learning_rate = 0.1   # yeni öğrenilen etkisi
discount_factor = 0.9 # şimdi mi önemli sonra mı 
epsilon = 0.2         # yeni deneyimler edinme oranı
epochs = 50           # tecrübe 

def move(state, action):
    row = state // area_size
    col = state % area_size

    if action == "up":
        row = max(row-1,0)
    elif action == "down":
        row = min(row+1, area_size -1)
    elif action == "left":
        col = max(col - 1, 0)
    elif action == "right":
        col = min(col+1, area_size -1)
    
    return row * area_size + col

for epoch in range(epochs):
    state = 0
    while state!=goal_state:
        if random.random() < epsilon:
            action = random.choice(actions)
            print(f"Random actinon: {action}")
        else: 
            max_q = max(q_table[state])
            best_indices = [i for i, q in enumerate(q_table[state]) if q == max_q]
            action_idx = random.choice(best_indices)  # aynı Q değeri varsa rastgele seç
            action = actions[action_idx]

        next_state = move(state, action)

        reward = 1 if next_state == goal_state else -0.1

        action_idx = actions.index(action)
        best_future = max(q_table[next_state])

        q_table[state][action_idx] += learning_rate * (
            reward + discount_factor * best_future - q_table[state][action_idx]
        )

        state = next_state
        
for i, row in enumerate(q_table):
    print(f"State {i}: ", end="")
    for action_name, q_val in zip(actions, row):
        print(f"{action_name}={q_val:.2f} ", end="")
    print()


