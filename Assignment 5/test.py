from optimization import Regressor

if __name__=="__main__":
    opt = Regressor()
    opt.fit(optimizer='adam', n_iters=100, render_animation=True)
