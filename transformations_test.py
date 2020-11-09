import main
import numpy as np

fails = []
rot_zeroes = main.rot_matrix([0,0,0])
rot_full = main.rot_matrix([2*np.pi,2*np.pi,2*np.pi])
rot_random = main.rot_matrix([0.766041154115559,0.274592004790069,0.855372572158561])
if rot_zeroes.all()!=np.array([[1,0,0],[0,1,0],[0,0,1]]).all():
    fails.append("Rot zeroes - %s"%rot_zeroes)
if rot_full.all()!=np.array([[0,0,0],[0,0,0],[0,0,0]]).all():
    fails.append("Rot full - %s"%rot_full)
if rot_random.all()!=np.array([[0.6936620, -0.6673143,  0.2711543],[0.6022519,  0.3308122, -0.7265369],[0.3951273,  0.6672742,  0.6313632]]).all():
    fails.append("Rot random - %s"%rot_random)

#Give up on automation, check the outputs
test_site=main.LaunchSite(1,0,0,0,0,0)
a=1000
b=5000
r_earth=6378137
"""
launch_to_inert_t0=main.pos_launch_to_inertial([0,0,0],test_site,0)
launch_to_inert_dir_test=main.pos_launch_to_inertial([a,0,b],test_site,0)
print(test_site)
print(launch_to_inert_t0)#Expected output is [r_earth,0,0]
print(launch_to_inert_dir_test)#[r_earth+b,0,-a]

test_site=main.LaunchSite(1,0,0,0,0,90)

launch_to_inert_t0=main.pos_launch_to_inertial([0,0,0],test_site,0)
launch_to_inert_dir_test=main.pos_launch_to_inertial([a,0,b],test_site,0)
print(test_site)
print(launch_to_inert_t0) #[0,0,r_earth]
print(launch_to_inert_dir_test) #[a,0,r_earth+b]

test_site=main.LaunchSite(1,0,0,0,0,45)
launch_to_inert_t0=main.pos_launch_to_inertial([0,0,0],test_site,0)
launch_to_inert_dir_test=main.pos_launch_to_inertial([a,0,b],test_site,0)
print(test_site)
print(launch_to_inert_t0)#[sqrt(r^2/2),0,sqrt(r^2/2)]
print(launch_to_inert_dir_test)#[sqrt(r^2/2),0,sqrt(r^2/2)] plus some trig stuff

test_site=main.LaunchSite(1,0,0,0,90,0)
launch_to_inert_t0=main.pos_launch_to_inertial([0,0,0],test_site,0)
launch_to_inert_dir_test=main.pos_launch_to_inertial([a,0,b],test_site,0)
print(test_site)
print(launch_to_inert_t0)#[0,r_earth,0]
print(launch_to_inert_dir_test)#[0,r_earth+b,-a] plus some trig stuff

#Now for some time dependance
test_site=main.LaunchSite(1,0,0,0,0,0)
launch_to_inert_t0=main.pos_launch_to_inertial([0,0,0],test_site,86164.09957374365/4)
launch_to_inert_dir_test=main.pos_launch_to_inertial([a,0,b],test_site,86164.09957374365/4)
print(test_site)
print(launch_to_inert_t0)#Results should be same as 90 degree longitude test
print(launch_to_inert_dir_test)

#And reverse
print(main.pos_inertial_to_launch([r_earth,0,0],test_site,0))#[0,0,0]
print(main.pos_inertial_to_launch([r_earth+b,0,-a],test_site,0))#[a,0,b]))

test_site=main.LaunchSite(1,0,0,0,0,90)
print(main.pos_inertial_to_launch([0,0,r_earth],test_site,0))#[0,0,0]
print(main.pos_inertial_to_launch([a,0,r_earth+b],test_site,0))#[a,0,b]


test_site=main.LaunchSite(1,0,0,0,0,0)
print(main.pos_inertial_to_launch([0,r_earth,0],test_site,86164.09957374365/4))#[0,0,0]
print(main.pos_inertial_to_launch([0,r_earth+b,-a],test_site,86164.09957374365/4))#[a,0,b]
"""
test_site=main.LaunchSite(1,0,0,0,0,0)
#print(main.vel_inertial_to_launch([0,main.ang_vel_earth*r_earth,0],test_site,0))#[0,0,0]
#print(main.vel_inertial_to_launch([10,main.ang_vel_earth*r_earth,10],test_site,0))#[10,0,-10]
test_site=main.LaunchSite(1,0,0,0,0,90)
#print(main.vel_inertial_to_launch([1039023,230923,2032930],test_site,0))#[1039023,230923,2032930]
test_site=main.LaunchSite(1,0,0,0,90,0)

#print(main.vel_inertial_to_launch([-main.ang_vel_earth*r_earth,0,0],test_site,0))#[0,0,0]
#print(main.vel_inertial_to_launch([a-main.ang_vel_earth*r_earth,b,0],test_site,0))#[b,-a,0]"""

test_site=main.LaunchSite(1,0,0,0,0,0)
##print(main.vel_launch_to_inertial([0,0,0],test_site,0))
#print(main.vel_launch_to_inertial([a,b,0],test_site,0))

test_site=main.LaunchSite(1,0,0,0,0,90)
#print(main.vel_launch_to_inertial([1039023,230923,2032930],test_site,0))#[1039023,230923,2032930]

test_site=main.LaunchSite(1,0,0,0,90,0)
print(main.vel_launch_to_inertial([0,0,0],test_site,0))#[0,456 ish,0]