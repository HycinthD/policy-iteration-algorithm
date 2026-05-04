# POLICY ITERATION ALGORITHM

## AIM
Implement policy iteration algorithm to find optimal policy by iteratively maximizing the value function.

## PROBLEM STATEMENT
The aim of this experiment is to find optimal policy for the mdp using policy iteration. Policy iteration includes policy evaluation and policy improvement where evaluation function is used to find optimal value function of each state and then improvement function is used to find best policy by comparing all the action value function as well as policy.

## POLICY ITERATION ALGORITHM
# Step 1:
Import required libraries.
# Step 2:
Load the frozen lake environment.
# Step 3:
Define the value evaluation, value improvement and value iteration functions.
# Step 4: 
Run the functions and display the results.

## POLICY IMPROVEMENT FUNCTION
### Name : Hycinth D
### Register Number : 212223240055
```python
def policy_improvement(V, P, gamma=1.0):
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))
    new_pi = lambda s: np.argmax(Q[s])
    return new_pi
```
## POLICY ITERATION FUNCTION
### Name : Hycinth D
### Register Number : 212223240055
```python
def policy_iteration(P, gamma=1.0, theta=1e-10):
    pi = np.zeros(len(P), dtype=int)
    while True:
        pi_func = lambda s: pi[s]
        V = policy_evaluation(pi_func, P, gamma, theta)
        policy_stable = True
        for s in range(len(P)):
            old_action = pi[s]
            Q = np.zeros(len(P[s]))
            for a in range(len(P[s])):
                for prob, next_state, reward, done in P[s][a]:
                    Q[a] += prob * (reward + gamma * V[next_state] * (not done))
            pi[s] = np.argmax(Q)
            if old_action != pi[s]:
                policy_stable = False
        if policy_stable:
            break

    return V, lambda s: pi[s]
```

## OUTPUT:
### 1. Policy, Value function and success rate for the Adversarial Policy
<img width="407" height="130" alt="image" src="https://github.com/user-attachments/assets/126d6476-65a7-4952-8fa8-41b45aa53976" />

<img width="387" height="110" alt="image" src="https://github.com/user-attachments/assets/fcba57b4-1df9-4d26-a365-8ed23fa7aad6" />

<img width="710" height="46" alt="image" src="https://github.com/user-attachments/assets/c96d9b20-fa12-465f-ac44-d1b6a951edfa" />


### 2. Policy, Value function and success rate for the Improved Policy
<img width="559" height="185" alt="image" src="https://github.com/user-attachments/assets/86be4446-a8d3-48e4-8ee6-095f8c0e4d3b" />

<img width="533" height="193" alt="image" src="https://github.com/user-attachments/assets/4a6dbeda-59c7-4503-9884-50fde304f05b" />

<img width="723" height="40" alt="image" src="https://github.com/user-attachments/assets/1b24d836-8687-45c1-be9d-c30deab60e05" />


### 3. Policy, Value function and success rate after policy iteration
<img width="521" height="161" alt="image" src="https://github.com/user-attachments/assets/4761e874-0e23-43a8-8469-c93e1f7b42f8" />

<img width="533" height="188" alt="image" src="https://github.com/user-attachments/assets/70669d45-3df0-4f22-93a7-d60d10b3df94" />

<img width="708" height="45" alt="image" src="https://github.com/user-attachments/assets/1241c103-d10f-4ed8-80a4-4ac9004519f6" />


## RESULT:
Therefore, policy iteration algorithm to find optimal policy by iteratively maximizing the value function is successfully implemented.
