import numpy as np

class KalmanFilter2D:
    def __init__(self, init_pos, init_vel):
        # State variables
        self.state = np.array([init_pos[0], init_vel[0], init_pos[1], init_vel[1]], dtype=float)

        # Covariance matrix
        self.P = np.identity(4) * 1e-6

        # State transition matrix
        self.F = np.array([[1, 1, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 1],
                           [0, 0, 0, 1]])

        # Measurement matrix
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])

        # Measurement noise covariance
        self.R = np.array([[50, 0],
                           [0, 50]])

        # Process noise covariance
        self.Q = np.array([[0.2, 0.1, 0, 0],
                          [0.1, 0.2, 0, 0],
                          [0, 0, 0.2, 0.1],
                          [0, 0, 0.1, 0.2]])

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.state

    def update(self, measurement):
        y = measurement - np.dot(self.H, self.state)
        
        # Dynamic measurement noise covariance
        scaling_factor = max(1, np.linalg.norm(y) / 10)
        dynamic_R = self.R * scaling_factor

        S = np.dot(self.H, np.dot(self.P, self.H.T)) + dynamic_R
        K = np.dot(self.P, np.dot(self.H.T, np.linalg.inv(S)))
        self.state = self.state + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))
