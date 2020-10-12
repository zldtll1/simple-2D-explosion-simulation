#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <math.h>
#include <algorithm>
#include "../lodepng/lodepng.h"

using namespace std;

class FluidQuantity {
	// Memory buffers for fluid quantity
	double *_src;
	double *_dst;

	// Width and Hight of simulation area 
	// or cell numbers in Width and Hight
	int _w;
	int _h;

	// X and Y offset from top left grid cell.
	// This is (0.5, 0.5) for centered quantities such as density,
	// and (0.0, 0.5) or (0.5, 0.0) for jittered quantities like the velocity.
	double _ox;
	double _oy;

	// Grid cell size
	double _hx;

	// Linear interpolate between a and b for x ranging from 0 to 1.
	// const成员函数用法详解: https://www.cnblogs.com/MakeView660/p/8446155.html
	double lerp(double a, double b, double x) const {
		return a + (b - a) * x;
	}

	// Simple forward Euler method for velocity integration in time.
	// Forward Euler: dX/dt = v --> X[t] = X[t+dt] - v[t] * dt.
	// Calculate the point's old position X[t] which is now in (x, y) use old velocity v[t].
	/* If use v[t+dt], this would be backward Euler. But that is impossible, since this function
	 * is the first step of Semi-Lagrangian which is used to get quantities[t+dt] include velocity.
	 * forward or backward Euler method is no difference, because it's very possible that both of them would introduce error.
	 *                  /t+dt
	 * X[t+dt] - X[t] = |    v(k)dk, 由拉格朗日中值定理知: 存在 t<= m <=t+dt, 使 X[t+dt] - X[t] = v[m] * dt.
	 *					/t
	 * 数值积分法原理及 matlab 程序实现: https://blog.csdn.net/qq_25829649/article/details/51166469
	 */
	void euler(double &x, double &y, double timestep, const FluidQuantity &u, const FluidQuantity &v) const {
		double uVel = u.lerp(x, y) / _hx;
		double vVel = v.lerp(x, y) / _hx;
		x -= uVel * timestep;
		y -= vVel * timestep;
	}

public:
	FluidQuantity(int w, int h, double ox, double oy, double hx)
		: _w(w), _h(h), _ox(ox), _oy(oy), _hx(hx) {
		_src = new double[_w*_h];
		_dst = new double[_w*_h];

		memset(_src, 0, _w*_h*sizeof(double));
	}

	~FluidQuantity() {
		delete[] _src;
		delete[] _dst;
	}

	void flip() {
		swap(_src, _dst);
	}

	const double *src() const {
		return _src;
	}

	// Read-only and read-write access to grid cells.
	double at(int x, int y) const {
		return _src[x + y * _w];
	}

	double &at(int x, int y) {
		return _src[x + y * _w];
	}

	// linear interpolate on grid at coordinate (x, y).
	// Coordinate will be clamped to lie in simulation domain.
	/* 1. 将坐标 (x, y) 转换到 quantity(u, v, d) 网格上的坐标, 需减去 offset,
	 *    即: x - _ox; y - _oy; 由于存在减法, 需保证结果不能为负(数组下标有效性),
	 *	  即: max(x - _ox, 0.0); max(y - _oy, 0.0); 
	 * 2. 需要取得待插值点周围的 4 个存有quantity量的整点的坐标, 对 (x, y) *舍弃小数位* 取得左上坐标,
	 *    即: int ix = (int)x; int iy = (int)y; 其他三点坐标通过加 1 获得.
	 *    此时, 要保证 ix + 1 or iy + 1 不能超过 quantity 网格的范围(数组下标有效性),
	 *    即: ix+1 <= _w-1 --> x <= ix+0.999 <= _w-1.001 --> x <= _w-1.001 (取1/1000精度),
	 *    因此: x = min(max(x - _ox, 0.0), _w - 1.001);
	 */
	double lerp(double x, double y) const {
		x = min(max(x - _ox, 0.0), _w - 1.001);
		y = min(max(y - _oy, 0.0), _h - 1.001);
		int ix = (int)x;
		int iy = (int)y;
		x -= ix;
		y -= iy;

		double x00 = at(ix + 0, iy + 0), x10 = at(ix + 1, iy + 0);
		double x01 = at(ix + 0, iy + 1), x11 = at(ix + 1, iy + 1);

		return lerp(lerp(x00, x10, x), lerp(x01, x11, x), y);
	}

	// Advect grid in velocity field u, v with given timestep.
	// ix、iy 为某 quantity 网格上的点位，通过增加偏置量 _ox、_oy 转为模拟区域的点位.
	// This function can advect every kind of quantities.  
	void advect(double timestep, const FluidQuantity &u, const FluidQuantity &v) {
		for (int iy = 0, idx = 0; iy < _h; iy++) {
			for (int ix = 0; ix < _w; ix++, idx++) {
				double x = ix + _ox;
				double y = iy + _oy;

				// First component: Integrate in time.
				euler(x, y, timestep, u, v);

				// Second component: Interpolate from grid.
				_dst[idx] = lerp(x, y);
			}
		}		
	}

	// Set fluid quantity inside the given rect to value 'v'.
	// fabs() 求实数的绝对值.
	void addInflow(double x0, double y0, double x1, double y1, double v) {
		int ix0 = (int)(x0 / _hx - _ox);
		int iy0 = (int)(y0 / _hx - _oy);
		int ix1 = (int)(x1 / _hx - _ox);
		int iy1 = (int)(y1 / _hx - _oy);

		for (int y = max(iy0, 0); y < min(iy1, _h); y++)
			for (int x = max(ix0, 0); x < min(ix1, _w); x++)
				if (fabs(_src[x + y * _w]) < fabs(v))
					_src[x + y * _w] = v;
	}
};

// Fluid solver class. Sets up the fluid quantities, forces incompressibility
// performs advection and adds inflows.
class FluidSolver {
	// Fluid quantities
	FluidQuantity *_d;
	FluidQuantity *_u;
	FluidQuantity *_v;

	// Width and height
	int _w;
	int _h;

	// Grid cell size and fluid density
	double _hx;
	double _density;

	// Arrays for:
	double *_r; // Right hand side of pressure solve
	double *_p; // Pressure solution

	// Build the pressure right hand side as the negative divergence
	//（a = F/m) : (u[t+dt] - u[t])/dt = -▽p/ρ --> ▽(u[t+dt] - u[t]) = -▽·▽p(dt/ρ)
	// 不可压缩 --> ▽u[t+dt] = 0 --> -▽u[t] = -▽·▽p(dt/ρ) 
	// 求解压力 p,建立形如 Ap = b 的矩阵：-(dt/ρ)▽·▽p = -▽u[t],
	// 即 Right hand side(Rhs) 为 -▽u[t].
	void buildRhs() {
		double scale = 1.0/_hx;

		for (int y = 0, idx = 0; y < _h; y++) {
			for (int x = 0; x < _w; x++, idx++) {
				_r[idx] = -scale * (_u->at(x + 1, y) - _u->at(x, y) +
									_v->at(x, y + 1) - _v->at(x, y));
			}
		}
	}

	// Perform the pressure solve using Gauss-Seidel.
	// The solver will run as long as it takes to get the relative error below
	// a threshold, but will never exceed 'limit' iterations.
	// 《Fluid Simulation for Computer Graphics 2nd》- Figure 5.5
	void project(int limit, double timestep) {
		double scale = timestep / (_density*_hx*_hx);

		double maxDelta;
		for (int iter = 0; iter < limit; iter++) {
			maxDelta = 0.0;
			for (int y = 0; y < _h; y++) {
				for (int x = 0; x < _w; x++) {
					int idx = x + y * _w;

					double  diag = 0.0, offDiag = 0.0;

					// Here we build the matrix implicitly as the five-point
					// stencil. Grid borders are assumed to be solid, i.e.
					// there is no fluid outside the simulaiton domain.
					// That is (0, )、(_W - 1, )、( , 0)、( , _h - 1) is solid.
					if (x > 0) {	 // [x-1, y] is fluid.
						diag    += scale;
						offDiag -= scale * _p[idx - 1];
					}
					if (y > 0) {	 // [x, y-1] is fluid.
						diag    += scale;
						offDiag -= scale * _p[idx - _w];
					}
					if (x < _w - 1) {// [x+1, y] is fluid.
						diag    += scale;
						offDiag -= scale * _p[idx + 1];
					}
					if (y < _h - 1) {// [x, y+1] is fluid.
						diag    += scale;
						offDiag -= scale * _p[idx + _w];
					}

					double newP = (_r[idx] - offDiag) / diag;
					maxDelta = max(maxDelta, fabs(_p[idx] - newP));
					_p[idx] = newP;
				}
			}

			if (maxDelta < 1e-5) {
				printf("Exiting solver after %d iterations, maximum change is %f\n", iter, maxDelta);
				return;
			}
		}

		printf("Exiting solver after %d iterations, maximum change is %f\n", limit, maxDelta);
	}

	// ####Add pressure to get explosion effect.
	void addPressure(double pressureX, double x0, double y0, double x1, double y1, double time, double duration) {
		if (time < duration) {
			int idx = (int)((x1 + x0) / (2.0 * _hx));
			int idy = (int)((y1 + y0) / (2.0 * _hx));
			int i = idx + idy * _w;
			//double oldp = _p[i];
			_p[i] *= pressureX;
			//printf("i: %d, oldP: %f, newP: %f\n", i, oldp, _p[i]);
		}
	}

	// Applies the computed pressure to the velocity field.
	void applyPressure(double timestep) {
		double scale = timestep / (_density*_hx);

		for (int y = 0, idx = 0; y < _h; y++)
			for (int x = 0; x < _w; x++, idx++) {
				_u->at(x,     y    ) -= scale * _p[idx];
				_u->at(x + 1, y    ) += scale * _p[idx];
				_v->at(x,     y    ) -= scale * _p[idx];
				_v->at(x,     y + 1) += scale * _p[idx];
			}
		for (int y = 0; y < _h; y++)
			_u->at(0, y) = _u->at(_w, y) = 0.0;
		for (int x = 0; x < _w; x++)
			_v->at(x, 0) = _v->at(x, _h) = 0.0;	
	}

public:
	FluidSolver(int w, int h, double density) : _w(w), _h(h), _density(density) {
		_hx = 1.0 / min(w, h);
		
		_d = new FluidQuantity(_w,     _h,     0.5, 0.5, _hx);
		_u = new FluidQuantity(_w + 1, _h,     0.0, 0.5, _hx);
		_v = new FluidQuantity(_w,     _h + 1, 0.5, 0.0, _hx);

		_r = new double[_w*_h];
		_p = new double[_w*_h];

		memset(_p, 0, _w*_h * sizeof(double));
	}

	~FluidSolver() {
		delete _d;
		delete _u;
		delete _v;

		delete[] _r;
		delete[] _p;
	}

	void update(double timestep) {
		buildRhs();
		project(600, timestep);
		applyPressure(timestep);

		_d->advect(timestep, *_u, *_v);
		_u->advect(timestep, *_u, *_v);
		_v->advect(timestep, *_u, *_v);

		// Make effect of advection visible, since it's not an in-place operation.
		_d->flip();
		_u->flip();
		_v->flip();
	}

	//#### Add pressure at source center before applyPressure().
	void updateWithExplosionEffect(double timestep, double pressureX, double x, double y, double w, double h, double time, double duration) {
		buildRhs();
		project(600, timestep);
		addPressure(pressureX, x, y, x+w, y+h, time, duration);
		applyPressure(timestep);

		_d->advect(timestep, *_u, *_v);
		_u->advect(timestep, *_u, *_v);
		_v->advect(timestep, *_u, *_v);

		/* Make effect of advection visible, since it's not an in-place operation */
		_d->flip();
		_u->flip();
		_v->flip();
	}

	// Set density and x/y velocity in given rectangle to d/u/v, respectively.
	void addInflow(double x, double y, double w, double h, double d, double u, double v) {
		_d->addInflow(x, y, x + w, y + h, d);
		_u->addInflow(x, y, x + w, y + h, u);
		_v->addInflow(x, y, x + w, y + h, v);
	}

	// Return the maximum allowed timestep. Note that the actual timestep
	// taken should usually be much below this to ensure accurate
	// simulation - just never above.
	double maxTimestep() {
		double maxVelocity = 0.0;
		for (int y = 0; y < _h; y++)
			for (int x = 0; x < _w; x++) {
				// Average velocity at grid cell center.
				double u = _u->lerp(x + 0.5, y + 0.5);
				double v = _v->lerp(x + 0.5, y + 0.5);

				double velocity = sqrt(u*u + v * v);
				maxVelocity = max(maxVelocity, velocity);
			}

		// Fluid should not flow more than two grid cells per iteration.
		double maxTimestep = 2.0 * _hx / maxVelocity;

		// Clamp to sensible maximum value in case of very small velocities.
		return min(maxTimestep, 1.0);
	}

	// Convert fluid density to RGBA image.
	void toImage(unsigned char *rgba) {
		for (int i = 0; i < _w*_h; i++) {
			int shade = (int)((1.0 - _d->src()[i]) * 255.0);// --> min density is 1.
			shade = max(min(shade, 255), 0);

			rgba[i * 4 + 0] = shade; // R
			rgba[i * 4 + 1] = shade; // G
			rgba[i * 4 + 2] = shade; // B
			rgba[i * 4 + 3] = 0xFF;  // A
		}
	}
};

int main() {
	// Play with these constants, if you want.
	const int sizeX = 128;
	const int sizeY = 128;

	const double density = 0.1;
	const double timestep = 0.005;

	unsigned char *image = new unsigned char[sizeX*sizeY * 4];

	FluidSolver *solver = new FluidSolver(sizeX, sizeY, density);

	double time = 0.0;
	int iterations = 0;

	//
	double pressureX = 0.0;
	double Vspeed = 0.0;
	double duration = 0.0;
	printf("Input \npressureX Vspeed duration\n");
	scanf("%lf %lf %lf", &pressureX, &Vspeed, &duration);
	//

	while (time < 8.0) {
		// Use four substeps per iteration.
		for (int i = 0; i < 4; i++) {
			solver->addInflow(0.4, 0.2, 0.2, 0.2, 1.0, 0.0, Vspeed);
			//solver->update(timestep);
			solver->updateWithExplosionEffect(timestep, pressureX, 0.4, 0.2, 0.2, 0.2, time, duration);
			time += timestep;
			fflush(stdout); // 清空缓冲区，强制输出到屏幕。
		}
		solver->toImage(image);

		char path[256];
		sprintf(path, "Frame%05d.png", iterations++);
		lodepng_encode32_file(path, image, sizeX, sizeY);
	}

	return 0;
}
