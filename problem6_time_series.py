import numpy as np

class ARIMA:
    def __init__(self, p=2, d=1, q=1, n_iterations=1000, learning_rate=0.001):
        self.p = p  # Auto-Regressive
        self.d = d  # Differencing
        self.q = q  # Moving Average
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.ar_params = None
        self.ma_params = None
        self.bias = 0

    def difference(self, series, d):
        diff = series.copy()
        for _ in range(d):
            diff = np.diff(diff)
        return diff

    def inverse_difference(self, original, diff, d):
        restored = diff.copy()
        for _ in range(d):
            restored = np.r_[original[:1], np.cumsum(restored) + original[0]]
        return restored

    def fit(self, series):
        series = np.array(series)
        diff_series = self.difference(series, self.d)
        n = len(diff_series)

        self.ar_params = np.zeros(self.p)
        self.ma_params = np.zeros(self.q)
        self.bias = 0
        errors = np.zeros(n)

        for _ in range(self.n_iterations):
            for t in range(max(self.p, self.q), n):
                ar_term = np.dot(self.ar_params, diff_series[t-self.p:t][::-1])
                ma_term = np.dot(self.ma_params, errors[t-self.q:t][::-1])
                y_pred = self.bias + ar_term + ma_term
                error = diff_series[t] - y_pred
                errors[t] = error

                # تحديث المعاملات
                self.bias += self.learning_rate * error
                self.ar_params += self.learning_rate * error * diff_series[t-self.p:t][::-1]
                self.ma_params += self.learning_rate * error * errors[t-self.q:t][::-1]

    def forecast(self, series, steps=5):
        series = np.array(series)
        diff_series = self.difference(series, self.d)
        n = len(diff_series)
        errors = np.zeros(n + steps)
        forecasts = []

        for t in range(n, n + steps):
            ar_term = np.dot(self.ar_params, diff_series[t-self.p:t][::-1])
            ma_term = np.dot(self.ma_params, errors[t-self.q:t][::-1])
            y_pred = self.bias + ar_term + ma_term
            forecasts.append(y_pred)
            errors[t] = 0  # نفترض أن الخطأ صفر في المستقبل
            diff_series = np.append(diff_series, y_pred)

        return self.inverse_difference(series, forecasts, self.d)



if __name__ == "__main__":
    # سلسلة زمنية بسيطة (مثلاً مبيعات شهرية)
    series = [100, 120, 130, 150, 170, 180, 200, 220, 240, 260]

    model = ARIMA(p=2, d=1, q=1, n_iterations=1000, learning_rate=0.001)
    model.fit(series)


    forecast = model.forecast(series, steps=5)
    print("Forecasted values:", np.round(forecast, 2))