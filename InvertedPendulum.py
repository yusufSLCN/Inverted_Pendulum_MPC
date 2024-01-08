### This file contains the drawing part for various systems. Given a state vector
### Will return an imageself.

import numpy as np
import cv2

class InvertedPendulum:
    def __init__(self, x_start, x_end, pendulum_len = 1):
        self.x_start = x_start
        self.x_end = x_end
        self.x_len = x_end - x_start

        self.ground_y = 450
        self.num_ground_points = 21
        self.window_size = (512, 1024, 3)
        self.m_to_pixel = self.window_size[1] / self.x_len 
        self.pendulum_len = pendulum_len * self.m_to_pixel


    def step( self, state_vec, t=None ):
        """ state vector :
                x0 : position of the cart
                x1 : veclocity of the cart
                x2 : angle of pendulum. In ref frame with x as forward of the cart and y as up. Angile with respect to ground plane
                x3 : angular velocity of the pendulum
        """
        cart_pos = state_vec[0]
        bob_ang  = state_vec[2]*180 / np.pi # degrees

        canvas = np.zeros(self.window_size, dtype='uint8' )

        # Ground line
        cv2.line(canvas, (0, self.ground_y), (canvas.shape[1], self.ground_y), (19,69,139), 4 )


        # Mark ground line
        for xd in np.linspace( self.x_start, self.x_end, self.num_ground_points):
            x = int((xd - self.x_start) * self.m_to_pixel)
            cv2.circle(canvas, (x, self.ground_y), 5, (0,255,0), -1)
            cv2.putText(canvas, f'{xd:.1f}', (x - 15, self.ground_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,250), 1)


        # Draw Wheels of the cart
        wheel_1_pos = int((cart_pos - 0.3 - self.x_start) * self.m_to_pixel)
        wheel_2_pos = int((cart_pos + 0.3 - self.x_start) * self.m_to_pixel)

        cv2.circle(canvas, (wheel_1_pos, self.ground_y - 30), 18, (255,255,255), 6 )
        cv2.circle(canvas, (wheel_2_pos, self.ground_y - 30), 18, (255,255,255), 6 )
        cv2.circle(canvas, (wheel_1_pos, self.ground_y - 30), 2, (255,255,255), -1 )
        cv2.circle(canvas, (wheel_2_pos, self.ground_y - 30), 2, (255,255,255), -1 )

        # Cart base
        cart_base_start = int((cart_pos - 0.5 - self.x_start) * self.m_to_pixel)
        cart_base_end   = int((cart_pos + 0.5 - self.x_start) * self.m_to_pixel)
        cart_base_y = 380
        cv2.line(canvas, (cart_base_start, cart_base_y), (cart_base_end, 380), (255,255,255), 6 )

        # Pendulum hinge
        pendulum_hinge_x = int((cart_pos - self.x_start) * self.m_to_pixel)
        pendulum_hinge_y = cart_base_y
        cv2.circle( canvas, (pendulum_hinge_x, pendulum_hinge_y), 10, (255,255,255), -1 )


        # Pendulum
        pendulum_bob_x = int( self.pendulum_len * np.cos( bob_ang / 180. * np.pi ) )
        pendulum_bob_y = int( self.pendulum_len * np.sin( bob_ang / 180. * np.pi ) )
        cv2.circle( canvas, (pendulum_hinge_x+pendulum_bob_x, pendulum_hinge_y-pendulum_bob_y), 10, (255,255,255), -1 )
        cv2.line( canvas, (pendulum_hinge_x, pendulum_hinge_y), (pendulum_hinge_x+pendulum_bob_x, pendulum_hinge_y-pendulum_bob_y), (255,255,255), 3 )

        # # Mark the current angle
        # angle_display = bob_ang % 360
        # if( angle_display > 180 ):
        #     angle_display = -360+angle_display
        # cv2.putText(canvas, f"theta= {angle_display:.1f} deg", (pendulum_hinge_x-15, pendulum_hinge_y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,250), 1);


        # Display on top
        if t is not None:
            cv2.putText(canvas, f"t= {t:.2f} sec", (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,250), 1)
            cv2.putText(canvas, f"theta= {bob_ang:.1f} deg", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,250), 1)
            cv2.putText(canvas, f"pos= {cart_pos:.2f} m", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,250), 1)

        return canvas


if __name__=="__main__":
    syst = InvertedPendulum()

    x = 0.
    sx = 1.
    theta = np.pi/3
    stheta = np.pi/3
    t = 0.
    while True:
        # x += sx*0.1
        # theta += stheta*1.
        # if( x > 5 ):
        #     sx = -1.
        # if( x < -5 ):
        #     sx = 1.0

        # theta = -9.8 / 1.5 * np.cos( t ) + 95.0 + 9.8/1.5

        rendered = syst.step( [x,0,theta,0] )
        cv2.imshow( 'im', rendered )

        if cv2.waitKey(30) == ord('q'):
            break

        t += 30./1000.
