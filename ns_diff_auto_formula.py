import torch
import numpa as np
#6/7 *rho_l *div(u tens u) = - grad(p) + div(2*mu_l* epsilon(u)) - ( (5* mu_l/2 H_t**2) + alpha(rho))u

def compute_pde_residual(self, input_int):
    
        input_int.requires_grad = True
        u = self.approximate_solution(input_int)

        # grad compute the gradient of a "SCALAR" function L with respect to some input nxm TENSOR Z=[[x1, y1],[x2,y2],[x3,y3],...,[xn,yn]], m=2
        # it returns grad_L = [[dL/dx1, dL/dy1],[dL/dx2, dL/dy2],[dL/dx3, dL/dy3],...,[dL/dxn, dL/dyn]]
        # Note: pytorch considers a tensor [u1, u2,u3, ... ,un] a vectorial function
        # whereas sum_u = u1 + u2 + u3 + u4 + ... + un as a "scalar" one

        # In our case ui = u(xi), therefore the line below returns:
        # grad_u = [[dsum_u/dx1, dsum_u/dy1],[dsum_u/dx2, dsum_u/dy2],[dsum_u/dx3, dL/dy3],...,[dsum_u/dxm, dsum_u/dyn]]
        # and dsum_u/dxi = d(u1 + u2 + u3 + u4 + ... + un)/dxi = d(u(x1) + u(x2) u3(x3) + u4(x4) + ... + u(xn))/dxi = dui/dxi
        grad_u = torch.autograd.grad(u.sum(), input_int, create_graph=True)[0]
        grad_u_x = grad_u[:, 0]
        grad_u_y = grad_u[:, 1]

        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_int[0], create_graph=True)[0][:, 1]
        grad_u_xy = torch.autograd.grad(grad_u_x.sum(), input_int[1], create_graph=True)[1][:, 1] 
        grad_u_yy = torch.autograd.grad(grad_u_y.sum(), input_int[1], create_graph=True)[1][:, 1]


        # div (u) = grad_u_x + grad_u_y
        
        div_term = 6/7 * rho_l *(np.tensordot(grad_u_x, u)+np.tensordot(u, grad_u_x)+np.tensordot(grad_u_y, u)+np.tensordot(u,grad_u_y))
        
        strain_term = 2/Re * ((1/2*np.array([grad_u_xx, grad_u_xy])+ np.array([grad_u_xx, grad_u_xy]).transpose())+
                            (1/2*np.array([grad_u_xy, grad_u_yy])+ np.array([grad_u_xy, grad_u_yy]).transpose()))
        
        drag_term = 1/Re * (5*L**2/2/H_t**2 + alpha(rho))* u
        
        residual = div_term + nabla_p - strain_term + drag_term
        
        
        return residual.reshape(-1, )
    
