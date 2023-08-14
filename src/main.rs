use macroquad::prelude::*;
use std::f32::consts::PI;

const VIEW_WIDTH: f32 = 1000.0;
const VIEW_HEIGHT: f32 = 1000.0;
const DAM_PARTICLES: usize = 250000;

const H: f32 = 16.0; // kernel radius
const EPS: f32 = H; // boundary epsilon
const HSQ: f32 = H * H; // radius^2
const MASS: f32 = 2.5; // assume all particles have the same mass
const GAS: f32 = 2000.0; // for equation of state
const REST_DENS: f32 = 300.0; // rest density
const VISC: f32 = 200.0; // viscosity constant
const G: Vec2 = Vec2::new(0.0, -10.0); // external (gravitational) forces
const DT: f32 = 0.0007; // integration timestep
const BOUND_DAMPING: f32 = -0.5;
const UPDATES_PER_FRAME: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq)]
struct Particle {
    x: Vec2,  // position
    v: Vec2,  // velocity
    f: Vec2,  // force
    rho: f32, // density
    p: f32,   // pressure
}

impl Particle {
    fn new(x: f32, y: f32) -> Self {
        Self {
            x: Vec2::new(x, y),
            v: Vec2::ZERO,
            f: Vec2::ZERO,
            rho: 0.0,
            p: 0.0,
        }
    }
}

struct Sim {
    particles: Vec<Particle>,
}

impl Sim {
    fn new() -> Self {
        Self { particles: vec![] }
    }

    fn clear(&mut self) {
        self.particles.clear();
    }

    fn init(&mut self) {
        let mut y = EPS;
        while y < VIEW_HEIGHT - EPS * 2.0 {
            let mut x = VIEW_WIDTH / 7.0;
            while x <= VIEW_WIDTH / 2.0 {
                if self.particles.len() < DAM_PARTICLES {
                    let jitter: f32 = rand::gen_range(0.0, 1.0);
                    self.particles.push(Particle::new(x + jitter, y));
                } else {
                    return;
                }
                x += H;
            }
            y += H;
        }
    }

    fn compute_density_pressure(&mut self) {
        let poly_6: f32 = 4.0 / (PI * H.powf(8.0));
        let x = self.particles.clone();
        self.particles.iter_mut().for_each(|pi| {
            pi.rho = 0.0;
            x.iter().for_each(|pj| {
                let rij = pj.x - pi.x;
                let r2: f32 = rij.dot(rij);
                if r2 < HSQ {
                    pi.rho += MASS * poly_6 * (HSQ - r2).powf(3.0);
                }
            });
            pi.p = GAS * (pi.rho - REST_DENS);
        });
    }

    fn compute_forces(&mut self) {
        let spiky_grad: f32 = -10.0 / (PI * H.powf(5.0));
        let visc_lap: f32 = 40.0 / (PI * H.powf(5.0));
        let x = self.particles.clone();
        self.particles.iter_mut().for_each(|pi| {
            let mut fpress: Vec2 = Vec2::ZERO;
            let mut fvisc: Vec2 = Vec2::ZERO;
            x.iter().for_each(|pj| {
                if pi != pj {
                    let rij = pj.x - pi.x;
                    let r: f32 = rij.dot(rij).sqrt();
                    if r < H {
                        fpress += -rij.normalize() * MASS * (pi.p + pj.p) / (2.0 * pj.rho)
                            * spiky_grad
                            * (H - r).powf(3.0);
                        fvisc += VISC * MASS * (pj.v - pi.v) / pj.rho * visc_lap * (H - r);
                    }
                }
            });
            let fgrav = G * MASS / pi.rho;
            pi.f = fpress + fvisc + fgrav;
        });
    }

    fn integrate(&mut self) {
        self.particles.iter_mut().for_each(|p| {
            p.v += DT * p.f / p.rho;
            p.x += DT * p.v;

            if p.x.x - EPS < 0.0 {
                p.v.x *= BOUND_DAMPING;
                p.x.x = EPS;
            }
            if p.x.x + EPS > VIEW_WIDTH {
                p.v.x *= BOUND_DAMPING;
                p.x.x = VIEW_WIDTH - EPS;
            }
            if p.x.y - EPS < 0.0 {
                p.v.y *= BOUND_DAMPING;
                p.x.y = EPS;
            }
            if p.x.y + EPS > VIEW_HEIGHT {
                p.v.y *= BOUND_DAMPING;
                p.x.y = VIEW_HEIGHT - EPS;
            }
        });
    }
}

#[macroquad::main("SPH")]
async fn main() {
    let mut sim = Sim::new();
    sim.init();

    loop {
        if is_key_pressed(KeyCode::Escape) {
            return;
        }
        if is_key_pressed(KeyCode::R) {
            sim.clear();
            sim.init();
        }

        for _ in 0..UPDATES_PER_FRAME {
            sim.compute_density_pressure();
            sim.compute_forces();
            sim.integrate();
        }

        let i = screen_width() / VIEW_WIDTH;
        let j = screen_height() / VIEW_HEIGHT;

        clear_background(GRAY);
        draw_text(&format!("{} FPS", get_fps()), 10.0, 30.0, 40.0, YELLOW);
        draw_text(
            &format!("{} PARTICLES", sim.particles.len()),
            10.0,
            60.0,
            40.0,
            YELLOW,
        );
        sim.particles.iter().for_each(|p| {
            draw_circle(
                p.x.x * i,
                screen_height() - p.x.y * j,
                4.0 * if i > j { i } else { j },
                BLUE,
            );
        });

        next_frame().await
    }
}
