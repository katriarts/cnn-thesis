import numpy as np
import pandas as pd

#save .npy file to .csv file
#save first 1000 rows

#basketball
basketball = np.load('full_numpy_bitmap_basketball.npy')
np.savetxt('basketball.csv', basketball, delimiter=',')
bbasketball = pd.read_csv('basketball.csv', nrows=5000)
np.savetxt('bbasketball.csv', bbasketball, delimiter=',')
#baseball
baseball = np.load('full_numpy_bitmap_baseball.npy')
np.savetxt('baseball.csv', baseball, delimiter=',')
bbaseball = pd.read_csv('baseball.csv', nrows=5000)
np.savetxt('bbaseball.csv', bbaseball, delimiter=',')
#shark
shark = np.load('full_numpy_bitmap_shark.npy')
np.savetxt('shark.csv', shark, delimiter=',')
sshark = pd.read_csv('shark.csv', nrows=5000)
np.savetxt('sshark.csv', sshark, delimiter=',')
#dolphin
dolphin = np.load('full_numpy_bitmap_dolphin.npy')
np.savetxt('dolphin.csv', dolphin, delimiter=',')
ddolphin = pd.read_csv('dolphin.csv', nrows=5000)
np.savetxt('ddolphin.csv', ddolphin, delimiter=',')
#duck
duck = np.load('full_numpy_bitmap_duck.npy')
np.savetxt('duck.csv', duck, delimiter=',')
dduck = pd.read_csv('duck.csv', nrows=5000)
np.savetxt('dduck.csv', dduck, delimiter=',')
#bird
bird = np.load('full_numpy_bitmap_bird.npy')
np.savetxt('bird.csv', bird, delimiter=',')
bbird = pd.read_csv('bird.csv', nrows=5000)
np.savetxt('bbird.csv', bbird, delimiter=',')
#van
van = np.load('full_numpy_bitmap_van.npy')
np.savetxt('van.csv', van, delimiter=',')
vvan = pd.read_csv('van.csv', nrows=5000)
np.savetxt('vvan.csv', vvan, delimiter=',')
#ambulance
ambulance = np.load('full_numpy_bitmap_ambulance.npy')
np.savetxt('ambulance.csv', ambulance, delimiter=',')
aambulance = pd.read_csv('ambulance.csv', nrows=5000)
np.savetxt('aambulance.csv', aambulance, delimiter=',')
#knee
knee = np.load('full_numpy_bitmap_knee.npy')
np.savetxt('knee.csv', knee, delimiter=',')
kknee = pd.read_csv('knee.csv', nrows=5000)
np.savetxt('kknee.csv', kknee, delimiter=',')
#leg
leg = np.load('full_numpy_bitmap_leg.npy')
np.savetxt('leg.csv', leg, delimiter=',')
lleg = pd.read_csv('leg.csv', nrows=5000)
np.savetxt('lleg.csv', lleg, delimiter=',')
