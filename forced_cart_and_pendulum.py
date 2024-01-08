import numpy as np
import cv2

from InvertedPendulum import InvertedPendulum
from CartModel import CartModel

from scipy.integrate import solve_ivp


def u( t ):
    t1 = 3
    t2 = 3.5

    if( t > t1 and t < t2 ):
        return 50.
    else:
        return 0.

# Pendulum and cart system. The motors on the cart turned at fixed time. In other words
# The motors are actuated to deliver forward x force from t=t1 to t=t2.
# Y : [ x, x_dot, theta, theta_dot]
def func3( t, y ):
    g = 9.8 # Gravitational Acceleration
    L = 1.5 # Length of pendulum

    m = 1.0 #mass of bob (kg)
    M = 5.0  # mass of cart (kg)


    x_ddot = u(t) - m*L*y[3]*y[3] * np.cos( y[2] ) + m*g*np.cos(y[2]) *  np.sin(y[2])
    x_ddot = x_ddot / ( M+m-m* np.sin(y[2])* np.sin(y[2]) )

    theta_ddot = -g/L * np.cos( y[2] ) - 1./L * np.sin( y[2] ) * x_ddot

    damping_theta =  - 0.5*y[3]
    damping_x =  - 1.0*y[1]

    return [ y[1], x_ddot + damping_x, y[3], theta_ddot + damping_theta ]


# Both cart and the pendulum can move.
if __name__=="__main__":
    # We need to write the Euler-lagrange equations for the both the
    # systems (bob and cart). The equations are complicated expressions. Although
    # it is possible to derive with hand. The entire notes are in media folder or the
    # blog post for this entry. Otherwse in essense it is very similar to free_fall_pendulum.py
    # For more comments see free_fall_pendulum.py
    cartModel = CartModel(L =1.5, m=1, M=5, g=9.8)
    sol = solve_ivp(cartModel.step, [0, 30], [ 0.0, 0., np.pi/2 - 0.1, 0. ], t_eval=np.linspace( 0, 20, 300))


    x_start = -5
    x_end = 5
    pendulum_len = cartModel.L
    syst = InvertedPendulum(x_start, x_end, pendulum_len)

    for i, t in enumerate(sol.t):
        rendered = syst.step( [sol.y[0,i], sol.y[1,i], sol.y[2,i], sol.y[3,i] ], t )
        cv2.imshow( 'im', rendered )
        cv2.moveWindow( 'im', 100, 100 )

        if cv2.waitKey(30) == ord('q'):
            break
