// Ceres Solver - A fast non-linear least squares minimizer
// Copyright 2010, 2011, 2012 Google Inc. All rights reserved.
// http://code.google.com/p/ceres-solver/
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// * Neither the name of Google Inc. nor the names of its contributors may be
//   used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Author: sameeragarwal@google.com (Sameer Agarwal),
//         dima@secretsauce.net     (Dima Kogan)
//
// Implementation of Powell's dogleg algorithm

#include "ceres/dogleg.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include <glog/logging.h>
#include "Eigen/Core"
#include "ceres/array_utils.h"
#include "ceres/evaluator.h"
#include "ceres/file.h"
#include "ceres/linear_least_squares_problems.h"
#include "ceres/linear_solver.h"
#include "ceres/matrix_proto.h"
#include "ceres/sparse_matrix.h"
#include "ceres/stringprintf.h"
#include "ceres/internal/eigen.h"
#include "ceres/internal/scoped_ptr.h"
#include "ceres/types.h"

namespace ceres {
namespace internal {
namespace {

// Numbers for clamping the size of the LM diagonal. The size of these
// numbers is heuristic. We will probably be adjusting them in the
// future based on more numerical experience. With jacobi scaling
// enabled, these numbers should be all but redundant.
const double kMinLevenbergMarquardtDiagonal = 1e-6;
const double kMaxLevenbergMarquardtDiagonal = 1e32;

// Small constant for various floating point issues.
const double kEpsilon = 1e-12;

// The Dogleg method is defined for full-rank JtJ. If a rank-deficient JtJ is
// encountered, I add a Levenberg-Marquardt-style mu*diag(JtJ) regularization
// contribution. Every time a rank-deficient JtJ is encountered, I geometrically
// increase mu. If mu gets too large, I give up and quit. Note that this logic
// is not a part of the solver algorithm itself, and is present JUST to handle
// singular JtJ matrices.
const double kMaxMu    = 1.0;
const double kMinMu    = 1e-10;
const double kMuFactor = 10.0;

// it is cheap to reject a too-large trust region, so I start with something
// "large". The solver will quickly move down to something reasonable. Only the
// user really knows if this is "large" or not, so this should be adjustable
const double kInitialTrustregion = 1.0e6;

// These are probably OK to leave alone. Tweaking them can maybe result in
// slightly faster convergence
const double kTrustregionDecreaseFactor    = 0.1;
const double kTrustregionIncreaseFactor    = 2.0;
const double kTrustregionIncreaseThreshold = 0.75;
const double kTrustregionDecreaseThreshold = 0.25;

struct Context
{
  const Minimizer::Options& options;
  Evaluator*                evaluator;
  LinearSolver*             linear_solver;
  IterationSummary&         iteration_summary;
  vector<int>&              iterations_to_dump;

  int iteration;

  // Jacobi scaling vector
  Vector scale;
  bool have_scale;

  Vector J_Jt_f, D, muD;

  // the cached update vectors. It's useful to cache these so that when a step is rejected, we can
  // reuse these when we retry
  Vector cauchy_step,       newton_step;
  double cauchy_step_norm2, newton_step_norm2;
  double cauchy_step_norm,  newton_step_norm;

  // Candidate step. This is on the trajectory from the origin point to the
  // cauchy point to the newton point
  Vector  dogleg_step_short; // used if the dogleg step is NOT a newton step
  Vector* dogleg_step;       // points to either newton_step or dogleg_step_short
#define dogleg_step_unscaled dogleg_step_short /* overlapping pre-allocated memory */

  Vector newton_minus_cauchy; // serves as pre-allocated memory
  Vector Jstep;               // serves as pre-allocated memory

  // whether the current update vectors are correct or not
  bool cauchy_step_valid, newton_step_valid;
  bool stepped_to_edge;

  double trustregion;
  double mu;

  double gradient_max_norm_0, gradient_tolerance;

  double total_cost, actual_cost_change, step_norm;


  Context(int num_effective_parameters, int num_residuals,
          Evaluator* _evaluator,
          LinearSolver* _linear_solver, IterationSummary& _iteration_summary,
          vector<int>& _iterations_to_dump, const Minimizer::Options& _options)
    : options(_options), evaluator(_evaluator), linear_solver(_linear_solver),
      iteration_summary(_iteration_summary),
      iterations_to_dump(_iterations_to_dump),
      scale(num_effective_parameters),
      J_Jt_f(num_residuals),
      D(num_effective_parameters), muD(num_effective_parameters),
      cauchy_step(num_effective_parameters), newton_step(num_effective_parameters),
      newton_minus_cauchy(num_effective_parameters), Jstep(num_residuals),
      dogleg_step_short(num_effective_parameters),
      trustregion(kInitialTrustregion), mu(0.0), have_scale(false)
  { }

  void invalidateCaches(void)
  {
    cauchy_step_valid = newton_step_valid = stepped_to_edge = false;
  }
};

struct OperatingPoint
{
  Vector                   x, f, Jt_f;
  double                   cost, model_cost;
  scoped_ptr<SparseMatrix> J;

  double x_norm;
  double gradient_max_norm;


  OperatingPoint(int num_parameters, int num_effective_parameters, int num_residuals,
                 SparseMatrix* J_data)
    : x(num_parameters), f(num_residuals), J(J_data),
      Jt_f(num_effective_parameters)
  { }

  // D = 1/sqrt(diag(J^T * J))
  void EstimateScale(double* D) {
    CHECK_NOTNULL(D);
    J->SquaredColumnNorm(D);
    for (int i = 0; i < J->num_cols(); ++i) {
      D[i] = 1.0 / (kEpsilon + sqrt(D[i]));
    }
  }

  bool computeOperatingPoint(Context& ctx)
  {
    f.setZero(); // is this necessary?

    if (!ctx.evaluator->Evaluate(x.data(), &cost, f.data(), J.get())) {
      LOG(WARNING) << "Failed to compute residuals and Jacobian. "
                   << "Terminating.";
      return false;
    }

    model_cost = f.squaredNorm() / 2.0;

    // estimate the Jacobian scale the first time
    if( ctx.options.jacobi_scaling )
    {
      /////////////// scaling ///////////////////////
      // The Dogleg method uses a sphere in state space for its trust region,
      // which implies even scaling for ALL the state variables. I thus compute
      // a vector of scale factors s, and run the dogleg solver as if the state
      // variables were thusly scaled. s has the same units as x, so x_scaled =
      // x/s is unitless. Let Ws = diag(s), and let all vectors be COLUMN
      // vectors
      // f = f(x) = f(x_scaled * s). J_scaled = df/dx_scaled = J Ws
      //
      // Everything the algorithm touches (other than the x vector itself) will
      // be done in the scaled-variable space. This is done because
      // num_effective_parameters != num_parameters
      if( !ctx.have_scale )
      {
        ctx.have_scale = true;
        EstimateScale( ctx.scale.data() );
      }
      J->ScaleColumns(ctx.scale.data());
    }

    x_norm = x.norm();

    // compute Jt*f
    Jt_f.setZero();
    J->LeftMultiply(f.data(), Jt_f.data());
    gradient_max_norm = Jt_f.lpNorm<Eigen::Infinity>();

    return true;
  }
};


// D = diag(J^T * J)
void LevenbergMarquardtDiagonal(const SparseMatrix& jacobian,
                                double* D) {
  CHECK_NOTNULL(D);
  jacobian.SquaredColumnNorm(D);
  for (int i = 0; i < jacobian.num_cols(); ++i) {
    D[i] = min(max(D[i], kMinLevenbergMarquardtDiagonal),
               kMaxLevenbergMarquardtDiagonal);
  }
}

bool RunCallback(IterationCallback* callback,
                 const IterationSummary& iteration_summary,
                 Solver::Summary* summary) {
  const CallbackReturnType status = (*callback)(iteration_summary);
  switch (status) {
    case SOLVER_TERMINATE_SUCCESSFULLY:
      summary->termination_type = USER_SUCCESS;
      VLOG(1) << "Terminating on USER_SUCCESS.";
      return false;
    case SOLVER_ABORT:
      summary->termination_type = USER_ABORT;
      VLOG(1) << "Terminating on USER_ABORT.";
      return false;
    case SOLVER_CONTINUE:
      return true;
    default:
      LOG(FATAL) << "Unknown status returned by callback: "
                 << status;
  }
}

bool checkGradientZero(const OperatingPoint& point, const Context& ctx)
{
  // Check if the starting point is an optimum.
  VLOG(2) << "Gradient max norm: " << point.gradient_max_norm
          << " tolerance: "        << ctx.gradient_tolerance
          << " ratio: "            << point.gradient_max_norm / ctx.gradient_max_norm_0
          << " tolerance: "        << ctx.options.gradient_tolerance;
  if (point.gradient_max_norm <= ctx.gradient_tolerance) {
    VLOG(1) << "Terminating on GRADIENT_TOLERANCE. "
            << "Relative gradient max norm: "
            << point.gradient_max_norm / ctx.gradient_max_norm_0
            << " <= " << ctx.options.gradient_tolerance;
    return true;
  }

  return false;
}


void computeCauchyUpdate(OperatingPoint& point, Context& ctx)
{
  // I already have this data, so don't need to recompute
  if(ctx.cauchy_step_valid)
    return;
  ctx.cauchy_step_valid = true;

  // I look at a step in the steepest direction that minimizes my
  // quadratic error function (Cauchy point). If this is past my trust region,
  // I move as far as the trust region allows along the steepest descent
  // direction. My error function is F=norm2(f(x)). dF/dx = 2*ft*J.
  // This is proportional to Jt_f, which is thus the steepest ascent direction.
  //
  // Thus along this direction we have F(k) = norm2(f(x + k Jt_f)). The Cauchy
  // point is where F(k) is at a minumum:
  // dF_dk = 2 f(x + k Jt_f)t  J Jt_f ~ (f + k J Jt_f)t J Jt_f =
  // = ft J Jt f + k ft J Jt J Jt f = norm2(Jt f) + k norm2(J Jt f)
  // dF_dk = 0 -> k= -norm2(Jt f) / norm2(J Jt f)
  // Summary:
  // the steepest direction is parallel to Jt*f. The Cauchy point is at
  // k*Jt*f where k = -norm2(Jt*f)/norm2(J*Jt*f)
  double norm2_Jt_f = point.Jt_f.squaredNorm();

  ctx.J_Jt_f.setZero();
  point.J->RightMultiply(point.Jt_f.data(), ctx.J_Jt_f.data());
  double norm2_J_Jt_f = ctx.J_Jt_f.squaredNorm();
  double k            = -norm2_Jt_f / norm2_J_Jt_f;

  ctx.cauchy_step = point.Jt_f.array() * k;
  ctx.cauchy_step_norm2 = k*k * norm2_Jt_f;

  ctx.cauchy_step_norm = sqrt(ctx.cauchy_step_norm2);

  VLOG(2) << "cauchy step size " << ctx.cauchy_step_norm;
}

bool computeGaussNewtonUpdate(OperatingPoint& point,
                              SolverTerminationType* result, Context& ctx)
{
  // I already have this data, so don't need to recompute
  if(ctx.newton_step_valid)
    return true;
  ctx.newton_step_valid = true;

  // try to solve the linear equation for the Gauss-Newton step. If JtJ is
  // singular, add a small constant to the diagonal. This constant gets larger
  // if we keep being singular
  LinearSolver::PerSolveOptions solve_options;
  solve_options.q_tolerance = ctx.options.eta;
  // Disable r_tolerance checking. Since we only care about
  // termination via the q_tolerance. As Nash and Sofer show,
  // r_tolerance based termination is essentially useless in
  // Truncated Newton methods.
  solve_options.r_tolerance = -1.0;
  while(true)
  {
    if( ctx.mu > 0.0 )
    {
      LevenbergMarquardtDiagonal(*point.J, ctx.D.data());

      ctx.muD = (ctx.mu * ctx.D).array().sqrt();
      solve_options.D = ctx.muD.data();
    }
    else
      solve_options.D = NULL;

    // Invalidate the output array newton_step, so that we can detect if
    // the linear solver generated numerical garbage.  This is known
    // to happen for the DENSE_QR and then DENSE_SCHUR solver when
    // the Jacobin is severly rank deficient and mu is too small.
    InvalidateArray(ctx.evaluator->NumEffectiveParameters(), ctx.newton_step.data());

    const time_t linear_solver_start_time = time(NULL);
    LinearSolver::Summary linear_solver_summary =
      ctx.linear_solver->Solve(point.J.get(),
                               point.f.data(),
                               solve_options,
                               ctx.newton_step.data());
    ctx.iteration_summary.linear_solver_time_sec   = time(NULL) - linear_solver_start_time;
    ctx.iteration_summary.linear_solver_iterations = linear_solver_summary.num_iterations;

    if (binary_search(ctx.iterations_to_dump.begin(),
                      ctx.iterations_to_dump.end(),
                      ctx.iteration)) {
      CHECK(DumpLinearLeastSquaresProblem(ctx.options.lsqp_dump_directory,
                                          ctx.iteration,
                                          ctx.options.lsqp_dump_format_type,
                                          point.J.get(),
                                          ctx.muD.data(),
                                          point.f.data(),
                                          ctx.newton_step.data(),
                                          ctx.options.num_eliminate_blocks))
        << "Tried writing linear least squares problem: "
        << ctx.options.lsqp_dump_directory
        << " but failed.";
    }

    // We ignore the case where the linear solver did not converge,
    // since the partial solution computed by it still maybe of use,
    // and there is no reason to ignore it, especially since we
    // spent so much time computing it.
    if ((linear_solver_summary.termination_type == TOLERANCE) ||
        (linear_solver_summary.termination_type == MAX_ITERATIONS))
    {
      if (!IsArrayValid(ctx.evaluator->NumEffectiveParameters(), ctx.newton_step.data())) {
        LOG(WARNING) << "Linear solver failure. Failed to compute a finite "
                     << "step. Terminating. Please report this to the Ceres "
                     << "Solver developers.";
        *result = NUMERICAL_FAILURE;
        return false;
      }

      break;
    }

    VLOG(1) << "Linear solver failure with mu == " << ctx.mu
            << ". retrying with a higher mu";

    // singular JtJ. Raise mu and go again
    if( ctx.mu == 0.0) ctx.mu  = kMinMu;
    else               ctx.mu *= kMuFactor;

    if( ctx.mu > kMaxMu )
    {
      VLOG(2) << "mu = " << ctx.mu
              << " is too large; giving up";
      *result = NUMERICAL_FAILURE;
      return false;
    }

    VLOG(2) << "singular JtJ. Adding " << ctx.mu
            << "*diag I from now on";
  }

  ctx.newton_step_norm2 = ctx.newton_step.squaredNorm();
  ctx.newton_step_norm  = sqrt( ctx.newton_step_norm2 );

  ctx.newton_step = -ctx.newton_step;

  VLOG(2) << "gauss-newton step size " << ctx.newton_step_norm;
  return true;
}

void computeInterpolatedUpdate(Vector& update, Context& ctx)
{
  // I interpolate between the Cauchy-point step and the Gauss-Newton step
  // to find a step that takes me to the edge of my trust region.
  //
  // I have something like norm2(a + k*(b-a)) = dsq
  // = norm2(a) + 2*at*(b-a) * k + norm2(b-a)*k^2 = dsq
  // let c = at*(b-a), l2 = norm2(b-a) ->
  // l2 k^2 + 2*c k + norm2(a)-dsq = 0
  //
  // This is a simple quadratic equation:
  // k = (-2*c +- sqrt(4*c*c - 4*l2*(norm2(a)-dsq)))/(2*l2)
  //   = (-c +- sqrt(c*c - l2*(norm2(a)-dsq)))/l2

  // to make 100% sure the discriminant is positive, I choose a to be the
  // cauchy step.  The solution must have k in [0,1], so I much have the
  // +sqrt side, since the other one is negative
  double dsq          = ctx.trustregion*ctx.trustregion;
  double norm2a       = ctx.cauchy_step_norm2;

  ctx.newton_minus_cauchy = ctx.newton_step.array() - ctx.cauchy_step.array();
  double l2               = ctx.newton_minus_cauchy.squaredNorm();
  double c                = ctx.newton_minus_cauchy.dot(ctx.cauchy_step);
  double discriminant     = c*c - l2* (norm2a - dsq);
  if(discriminant < 0.0)
  {
    VLOG(2) << "Interpolated dogleg step: negative discriminant " << discriminant;
    discriminant = 0.0;
  }
  double k = (-c + sqrt(discriminant))/l2;

  update = ctx.cauchy_step.array() + k*ctx.newton_minus_cauchy.array();
}

double computeExpectedImprovement(OperatingPoint& point, Context& ctx)
{
  // My error function is F=norm2(f(x + step))/2. F(0) - F(step) =
  // = 1/2 (norm2(f) - norm2(f + J*step)) =
  // = -inner(f,J*step) - 1/2*norm2(J*step)
  // = -inner(Jt_f,step) - 1/2*norm2(J*step)
  ctx.Jstep.setZero();
  point.J->RightMultiply(ctx.dogleg_step->data(), ctx.Jstep.data());

  // max() to handle a potential round-off error
  return
    max(kEpsilon, -point.Jt_f.dot(*ctx.dogleg_step) - 0.5*ctx.Jstep.squaredNorm());
}

// I have a candidate step. I adjust the trustregion accordingly, and also
// report whether this step should be accepted (0 == rejected, otherwise
// accepted)
bool evaluateStep_adjustTrustRegion(const OperatingPoint& before,
                                    const OperatingPoint& after,
                                    double expectedImprovement,
                                    Context& ctx)
{
  double observedImprovement = before.model_cost - after.model_cost;

  double rho = observedImprovement / expectedImprovement;
  VLOG(2) << "[Model cost] expected improvement: " << expectedImprovement
          << ", got improvement "                  << observedImprovement
          << ". rho = " << rho;

  if(rho < kTrustregionDecreaseThreshold)
  {
    VLOG(2) << "rho too small. decreasing trust region";

    // Our model doesn't fit well. We should reduce the trust region size. If
    // the trust region size was affecting the attempted step, do this by a
    // constant factor. Otherwise, drop the trustregion to attempted step size
    // first
    if( !ctx.stepped_to_edge )
      ctx.trustregion = ctx.newton_step_norm;

    ctx.trustregion *= kTrustregionDecreaseFactor;
  }
  else if (rho > kTrustregionIncreaseThreshold && ctx.stepped_to_edge)
  {
    VLOG(2) << "rho large enough. increasing trust region";
    ctx.trustregion *= kTrustregionIncreaseFactor;
  }

  return rho > 0.0;
}

// takes a step from the given operating point, using the given trust region
// radius. Returns the expected improvement, based on the step taken and the
// linearized x(p). If we can stop iterating, returns a negative number
double takeStep(OperatingPoint& from, Vector& x_new,
                SolverTerminationType* result, Context& ctx)
{
  VLOG(2) << "Trying a step with trustregion " << ctx.trustregion;

  computeCauchyUpdate(from, ctx);

  if(ctx.cauchy_step_norm2 >= ctx.trustregion*ctx.trustregion)
  {
    VLOG(2) << "taking cauchy step";

    // cauchy step goes beyond my trust region, so I do a gradient descent
    // to the edge of my trust region and call it good
    ctx.dogleg_step_short = ctx.cauchy_step * ctx.trustregion / ctx.cauchy_step_norm;
    ctx.dogleg_step = &ctx.dogleg_step_short;

    ctx.stepped_to_edge = true;
  }
  else
  {
    // I'm not yet done. The cauchy point is within the trust region, so I can
    // go further. I look at the full Gauss-Newton step. If this is within the
    // trust region, I use it. Otherwise, I find the point at the edge of my
    // trust region that lies on a straight line between the Cauchy point and
    // the Gauss-Newton solution, and use that. This is the heart of Powell's
    // dog-leg algorithm.
    if( !computeGaussNewtonUpdate(from, result, ctx) )
      return -1.0;

    if(ctx.newton_step_norm2 <= ctx.trustregion*ctx.trustregion)
    {
      VLOG(2) << "taking GN step";

      // full Gauss-Newton step lies within my trust region. Take the full step
      ctx.dogleg_step = &ctx.newton_step;

      ctx.stepped_to_edge = false;
    }
    else
    {
      VLOG(2) << "taking interpolated step";

      // full Gauss-Newton step lies outside my trust region, so I interpolate
      // between the Cauchy-point step and the Gauss-Newton step to find a step
      // that takes me to the edge of my trust region.
      computeInterpolatedUpdate(ctx.dogleg_step_short, ctx);
      ctx.dogleg_step = &ctx.dogleg_step_short;

      ctx.stepped_to_edge = true;
    }
  }

  // Check step length based convergence. If the step length is
  // too small, then we are done.
  const double step_size_tolerance = ctx.options.parameter_tolerance *
    (from.x_norm + ctx.options.parameter_tolerance);

  ctx.dogleg_step_unscaled = ctx.dogleg_step->array() * ctx.scale.array();
  double step_norm = ctx.dogleg_step_unscaled.norm();

  VLOG(2) << "Step size: "  << step_norm
          << " tolerance: " << step_size_tolerance
          << " ratio: "     << step_norm / step_size_tolerance
          << " tolerance: " << ctx.options.parameter_tolerance;
  if (step_norm <= ctx.options.parameter_tolerance *
      (from.x_norm + ctx.options.parameter_tolerance)) {
    *result = PARAMETER_TOLERANCE;
    VLOG(1) << "Terminating on PARAMETER_TOLERANCE."
            << "Relative step size: " << step_norm / step_size_tolerance
            << " <= "                 << ctx.options.parameter_tolerance;
    return -1.0;
  }


  // take the step
  if (!ctx.evaluator->Plus(from.x.data(), ctx.dogleg_step_unscaled.data(),
                           x_new.data())) {
    LOG(WARNING) << "Failed to compute Plus(x, delta, x_plus_delta). "
                 << "Terminating.";
    *result = NUMERICAL_FAILURE;
    return -1.0;
  }

  return computeExpectedImprovement(from, ctx);
}

void storeSolution(double* x_out, int N, OperatingPoint& point)
{
  memcpy(x_out, point.x.data(), N * sizeof(*x_out) );
}
}  // namespace

Dogleg::~Dogleg() {}

void Dogleg::Minimize(const Minimizer::Options& options,
                      Evaluator* evaluator,
                      LinearSolver* linear_solver,
                      const double* initial_parameters,
                      double* final_parameters,
                      Solver::Summary* summary) {

  time_t    start_time               = time(NULL);
  const int num_parameters           = evaluator->NumParameters();
  const int num_effective_parameters = evaluator->NumEffectiveParameters();
  const int num_residuals            = evaluator->NumResiduals();

  summary->termination_type       = NO_CONVERGENCE;
  summary->num_successful_steps   = 0;
  summary->num_unsuccessful_steps = 0;

  // Ask the Evaluator to create the jacobian matrix. The sparsity
  // pattern of this matrix is going to remain constant, so we only do
  // this once and then re-use this matrix for all subsequent Jacobian
  // computations.
  scoped_ptr<OperatingPoint>
    before_step( new OperatingPoint(num_parameters, num_effective_parameters, num_residuals,
                                    evaluator->CreateJacobian()) );
  scoped_ptr<OperatingPoint>
    after_step ( new OperatingPoint(num_parameters, num_effective_parameters, num_residuals,
                                    evaluator->CreateJacobian()) );
  // store the seed into before_step
  memcpy(before_step->x.data(), initial_parameters,
         num_parameters * sizeof(*initial_parameters) );


  IterationSummary iteration_summary;
  vector<int>      iterations_to_dump = options.lsqp_iterations_to_dump;
  Context ctx(num_effective_parameters, num_residuals,

              evaluator, linear_solver,
              iteration_summary, iterations_to_dump, options);

  if( !before_step->computeOperatingPoint(ctx) )
  {
    summary->termination_type = NUMERICAL_FAILURE;
    return;
  }
  // I just got a new operating point, so the current update vectors aren't
  // valid anymore, and should be recomputed, as needed
  ctx.invalidateCaches();


  // This is a poor way to do this computation. Even if fixed_cost is
  // zero, because we are subtracting two possibly large numbers, we
  // are depending on exact cancellation to give us a zero here. But
  // initial_cost and cost have been computed by two different
  // evaluators. One which runs on the whole problem (in
  // solver_impl.cc) in single threaded mode and another which runs
  // here on the reduced problem, so fixed_cost can (and does) contain
  // some numerical garbage with a relative magnitude of 1e-14.
  //
  // The right way to do this, would be to compute the fixed cost on
  // just the set of residual blocks which are held constant and were
  // removed from the original problem when the reduced problem was
  // constructed.
  summary->fixed_cost = summary->initial_cost - before_step->cost;

  ctx.total_cost         = summary->fixed_cost + before_step->cost;
  ctx.actual_cost_change = 0.0;
  ctx.step_norm          = 0.0;


  // We need the max here to guard againt the gradient being zero.
  ctx.gradient_max_norm_0 = max(before_step->gradient_max_norm, kEpsilon);
  ctx.gradient_tolerance = options.gradient_tolerance * ctx.gradient_max_norm_0;


  ctx.iteration = 0;

  // Parse the iterations for which to dump the linear problem.
  sort(iterations_to_dump.begin(), iterations_to_dump.end());

  iteration_summary.iteration                = ctx.iteration;
  iteration_summary.step_is_successful       = false;
  iteration_summary.cost                     = ctx.total_cost;
  iteration_summary.cost_change              = ctx.actual_cost_change;
  iteration_summary.gradient_max_norm        = before_step->gradient_max_norm;
  iteration_summary.step_norm                = ctx.step_norm;
  iteration_summary.relative_decrease        = 0.0;
  iteration_summary.mu                       = ctx.mu;
  iteration_summary.trustregion              = ctx.trustregion;
  iteration_summary.eta                      = options.eta;
  iteration_summary.linear_solver_iterations = 0;
  iteration_summary.linear_solver_time_sec   = 0.0;
  iteration_summary.iteration_time_sec       = (time(NULL) - start_time);
  if (options.logging_type >= PER_MINIMIZER_ITERATION) {
    summary->iterations.push_back(iteration_summary);
  }

  if( checkGradientZero(*before_step, ctx) )
  {
    summary->termination_type = GRADIENT_TOLERANCE;

    storeSolution(final_parameters, num_parameters, *before_step);
    return;
  }

  // Call the various callbacks.
  for (int i = 0; i < options.callbacks.size(); ++i) {
    if (!RunCallback(options.callbacks[i], iteration_summary, summary)) {
      storeSolution(final_parameters, num_parameters, *before_step);
      return;
    }
  }

  while ((ctx.iteration < options.max_num_iterations) &&
         (time(NULL) - start_time) <= options.max_solver_time_sec)
  {
    VLOG(2) << ""; // to visibly break up the steps in the log

    time_t iteration_start_time = time(NULL);

    while(true)
    {
      SolverTerminationType stepResult;
      double expected_improvement =
        takeStep(*before_step, after_step->x, &stepResult, ctx);

      // negative expectedImprovement is used to indicate that we're done
      if(expected_improvement < 0.0)
      {
        summary->termination_type = stepResult;
        return;
      }

      if( !after_step->computeOperatingPoint(ctx) )
      {
        summary->termination_type = NUMERICAL_FAILURE;
        return;
      }

      if( checkGradientZero(*after_step, ctx) )
      {
        summary->termination_type = GRADIENT_TOLERANCE;
        storeSolution(final_parameters, num_parameters, *after_step);
        return;
      }


      // Check function value based convergence.
      ctx.actual_cost_change = before_step->cost - after_step->cost;

      VLOG(2) << "[Nonlinear cost] current: " << before_step->cost
              << " new : "                    << after_step->cost
              << " change: "                  << ctx.actual_cost_change
              << " relative change: "         << fabs(ctx.actual_cost_change) / before_step->cost
              << " tolerance: "               << options.function_tolerance;
      if (fabs(ctx.actual_cost_change) < options.function_tolerance * after_step->cost) {
        VLOG(1) << "Termination on FUNCTION_TOLERANCE."
                << " Relative cost change: " << fabs(ctx.actual_cost_change) / after_step->cost
                << " tolerance: " << options.function_tolerance;
        summary->termination_type = FUNCTION_TOLERANCE;
        storeSolution(final_parameters, num_parameters, *after_step);
        return;
      }


      if( evaluateStep_adjustTrustRegion(*before_step, *after_step, expected_improvement, ctx) )
      {
        VLOG(2) << "accepted step";

        // I accept this step, so the after-step operating point is the before-step operating point
        // of the next iteration. I exchange the before- and after-step structures so that all the
        // pointers are still around and I don't have to re-allocate
        swap(before_step, after_step);
        ctx.invalidateCaches();
        break;
      }

      VLOG(2) << "rejected step";

      // This step was rejected. check if the new trust region size is small
      // enough to give up
      if(ctx.trustregion < options.trustregion_tolerance)
      {
        summary->termination_type = TRUSTREGION_TOLERANCE;
        VLOG(2) << "trust region small enough. Giving up. Done iterating!";

        storeSolution(final_parameters, num_parameters, *after_step);
        return;
      }

      // I have rejected this step, so I try again with the new trust region
    }

    // I just took a successful step
    ++summary->num_successful_steps;
    ++ctx.iteration;

    ctx.total_cost = summary->fixed_cost + before_step->cost;

    iteration_summary.iteration          = ctx.iteration;
    iteration_summary.step_is_successful = true;
    iteration_summary.cost               = ctx.total_cost;
    iteration_summary.cost_change        = ctx.actual_cost_change;
    iteration_summary.gradient_max_norm  = before_step->gradient_max_norm;
    iteration_summary.step_norm          = ctx.step_norm;
    iteration_summary.relative_decrease  = 0.0;
    iteration_summary.mu                 = ctx.mu;
    iteration_summary.trustregion        = ctx.trustregion;
    iteration_summary.eta                = options.eta;
    iteration_summary.iteration_time_sec = (time(NULL) - iteration_start_time);

    if (options.logging_type >= PER_MINIMIZER_ITERATION) {
      summary->iterations.push_back(iteration_summary);
    }

    // Call the various callbacks.
    for (int i = 0; i < options.callbacks.size(); ++i) {
      if (!RunCallback(options.callbacks[i], iteration_summary, summary)) {
        storeSolution(final_parameters, num_parameters, *before_step);
        return;
      }
    }
  }

  storeSolution(final_parameters, num_parameters, *before_step);
}

}  // namespace internal
}  // namespace ceres
