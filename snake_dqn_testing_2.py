# from old.environment import *
from environment_2 import *
from agents import *
import os
import datetime


##############################
# Game and Training parameters
##############################
wall_size = 7

screen_width = 70 + 2*wall_size
screen_height = 70 + 2*wall_size

snake_size = 7

log_freq = 50

nb_episodes = 30000
steps = 2000

counter = 0

c2 = 500

###############
# global param
###############

reward_list = []
score_list = []
steps_list = []
t_r = 0
tot_reward = []

best_score = 0
eps_val = 0.0
###############

env = SnakeEnvironment_2(screen_width, screen_height, snake_size, wall_size)
agent = DQNagent(4, env.states_space.shape, 10000, 64)

agent.model_policy.summary()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/tensorboard/' + current_time + '/train'
train_summary_writer = tf.summary.create_file_writer(train_log_dir)

loss = 0.0
os.makedirs("model/best",exist_ok=True)
for ep in range(nb_episodes):
	env, epd, disp, step = simulate(env, agent, log_freq, ep, steps, score_list, steps_list, counter, nb_episodes, t_r, c2)

	if ep > agent.batch_size:
		loss = agent.optimize_per()

	if env.score > best_score:
		best_score = env.score
		agent.save('best')
		eps_val = agent.current_eps

	if ep % log_freq == 0:
		with train_summary_writer.as_default():
			tf.summary.scalar('loss', loss , step=ep)
			tf.summary.scalar('steps', step, step=ep)
			tf.summary.scalar('reward', np.sum(epd['reward']), step=ep)
			tf.summary.scalar('score', env.score, step=ep)
			video = np.expand_dims(np.array(disp), 0)
			video_summary('dummy_snake', video, step=ep)


agent.plot_loss()

plt.plot(tot_reward)
plt.title('total reward evolution')
plt.ylabel('total reward')
plt.xlabel('episodes')
plt.show()

# plt.plot(reward_list)
# plt.title('episode reward evolution')
# plt.ylabel('reward per ep')
# plt.show()

plt.plot(steps_list)
plt.title('steps per ep')
plt.ylabel('steps')
plt.xlabel('ep')
plt.show()

plt.plot(score_list)
plt.title('score evolution')
plt.ylabel('final score')
plt.xlabel('game')
plt.show()

agent.save()