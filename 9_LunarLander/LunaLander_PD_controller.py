import gymnasium as gym
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
import numpy as np


def pid(state, params):
    """
    расчет управляющего воздействия на основе ПД-регуляторования
    :param state: состояния ОУ
    :param params: параметры ПД-регуляторов
    :return: управляющее воздействие
    """

    # Коэффициенты ПД-регулятора
    kp_alt = params[0]  # пропорциональная состовляющая по x
    kd_alt = params[1]  # дифференцирующая состовляющая по x
    kp_ang = params[2]  # пропорциональная состовляющая по углу
    kd_ang = params[3]  # дифференцирующая состовляющая по углу

    # расчет целевой переменной
    alt_tgt = np.abs(state[0])
    ang_tgt = (.25 * np.pi) * (state[0] + state[2])

    # расчет ошибки
    alt_error = (alt_tgt - state[1])
    ang_error = (ang_tgt - state[4])

    # Формируем управляющее воздействие ПД-регулятора
    alt_adj = kp_alt * alt_error + kd_alt * state[3]
    ang_adj = kp_ang * ang_error + kd_ang * state[5]


    # Приводим к интервалу (-1,  1)
    a = np.array([alt_adj, ang_adj])
    a = np.clip(a, -1, +1)

    # Если есть точка соприкосновения с землей, то глушим двигатели, никакие действия не пердаем
    if state[6] or state[7]:
        a[:] = 0
    return a


def start_game(environment, params, video_recorder=False):
    """
    Симуляция
    :param environment: среда Gym
    :param params: параметры ПД-регулятора
    :param video_recorder: объект для записи видео. False - без записи видео
    :return: суммарное качество посадки
    """
    state, _ = environment.reset()
    done = False
    total = 0
    while not done:
        environment.render()
        if video_recorder:
            video_recorder.capture_frame()

        # случайное действие
        # action = env.action_space.sample()

        # ПД-регулятор
        action = pid(state, params)
        state, reward, done, info, _ = environment.step(action)
        total += reward

        # print(state)  # ‘x’: 10 ‘y’: 6.666 ‘vx’: 5
        # ‘vy’: 7.5 ‘angle’: 1 ‘angular velocity’: 2.5

        # print(reward, done, info, action)
    return total


def optimize(params, current_score, env, step):
    """
    Подбор парамтеров
    :param params: стартовые параметры
    :param current_score: текущее качество посадки
    :param env: среда gym
    :param step: шаг оптимизации
    :return: параметры и качество
    """

    # добавить шум (меньше шума при увеличении n_steps)
    test_params = params + np.random.normal(0, 2 / step, size=params.shape)

    # тестирование параметров
    scores = []
    for trial in range(5):
        score = start_game(env, test_params)
        scores.append(score)
    avg = np.mean(scores)

    # Обновить параметры, если среднее значение награды
    # лучше чем с предыдущими параметрами
    if avg > current_score:
        return test_params, avg
    else:
        return params, current_score


if __name__ == "__main__":
    env_name = 'LunarLander-v2'

    env = gym.make(env_name,
                   render_mode="rgb_array",
                   continuous=True)

    print('Размер вектора состояния ОУ: ', env.observation_space.shape)
    print('Структура управляющего воздействия', env.action_space)

    optimize_params = False  # True - если хотим подобрать новые параметры
    params_pd = np.array([0.84827712, -1.55060286, -1.82178159, 0.8182234])

    if optimize_params:
        score = start_game(env, params_pd, video_recorder=False)
        for steps in range(100):
            params_pd, score = optimize(params_pd, score, env, steps+1)
            print("Step:", steps, "Score:", score, "Params:", params_pd)
    else:
        vid = VideoRecorder(env, path=f"random_luna_lander.mp4")
        params_pd = np.array([0.84827712, -1.55060286, -1.82178159, 0.8182234])
        score = start_game(env, params_pd, video_recorder=vid)

        vid.close()

    env.close()

