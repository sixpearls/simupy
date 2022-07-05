The goal with Version 2.0 is to improve consistency of the API with a strong mapping to mathematical concepts, within SimuPy, and with external libraries. This roadmap may be updated with more and more specific goals; I will try to maintain links to issues and PR's implementing them.

De-prioritize symbolics
    - Add codegen helpers to make it easier to generate numeric code to construct ``DynamicalSystem``\s from SymPy expressions
        - documentation, including symbolic (lambdify and codegen) specific examples
        - tests
        - Template for generating a class with parameters as well; possibly include as many extra features as possible (broadcastable; state/input and parameter jacobians; etc.) 
        - For broadcastable, need to handle unraveling broadcastable function calls correctly,
        - As human readable as possible without hurting performance
        - Templates for Cythonized, numba, etc
    - Remove symbolics from examples where easy (i.e., keeping symbolics for the DARE matrix ODE example would make sense)
    - Restructure library to de-emphasize symbolics as needed/possible
    - SymPy >= 1.5 compatiblity, probably use np.array's of SymPy objects/expressions rather than ``sp.Array``\'s since 

``DynamicalSystem`` interface changes
    - Instead of limiting the events to a single signal, it should provide a ``dim_event`` property and the ``event_function`` should provide a ``(..., dim_event)``-shaped array of the event. Update stiction example.
    - Add optional state and input jacobians to interface. 
    - Remove ``_equation`` from function names?
    - Metadata flag to indicate if functions are complex safe? No, can provide helper for creating jacobians using complex safe. 
    - Metadata flag to indicate broadcastable? Something like that. Need to determine best signature (..., n) vs (n, ...)
    - Helper to construct jacobians using numerical difference? how to estimate step size? Default to 1? Can we assume origin is in domain? Use initial condition and/or value to guess step size? Maybe an algorithm exists for this. And to use JAX or other differentiation methods??
    - Syntax candy for constructing systems?
        - Class method to cast any namespace into a ``DynamicalSystem``: If ``dim_event``, ``dim_input``, or ``dim_state`` is not assigned, set to 0; if ``dim_output`` not assigned, set to ``dim_state``, etc. numerical difference default partial derivative evaluator?
        - a Metaclass that can handle parameters etc in similar manner to how the symbolic codegen works? Or maybe the codegen just writes code that subclasses this meta? Part of it might be that annotated dataclass or whatever. Below is sketch of user code:

            .. code :: python 

                @jacobians_from_complex_step # this decorator adds state_jacobian(parameters, t, x, u) using complex step; otherwise may optionally provide own derivatives
                class MySystemModel(DynamicalSystem):
                    """
                    default repr prints as MySystemModel(l=<value>, m=<value>, b=<value>) which would be really pretty

                    """
                    dim_state=4 # how to handle dynamic state/input/output in this context? maybe that's not what this is used for? Then what is right way to do that? 
                    dim_input=1
                    dim_output=4
                    l: float # length of pendulum
                    g=9.8 # m/s^2 gravity constant; use equality (class attribute) for non-parametric constant -- can this be extended for aero module of Vehicle? would need base plus increments
                    #  dataclasses have a flag for indicating a paramete doens't go to repr; there may be a conditional version
                    m: float # mass
                    b: float # damping??
                    default_initial_condition = np.zeros(4) # default initialcondition; default default is also zeros?

                    @vectorized # annotation must just add an is_vectorized flag to the function that BlockDiagrams use??
                    # should be able to use numba or something here desired
                    def state_equation_function(parameters, t, x, u):
                        """
                        I would kind o flike it if the signature did not include self and the parameters got magically applied to the namespace but I don't think it works that way
                        but, we can pretend it's just "parameters"
                        For some reason I want to use assume shapes for t: ([1], ...), x: (dim_state, ...), u: (dim_input, ...)
                        it seems reasonable to assume that a naive/pure python unpacking on state/input would work! Even though rows (0th axis) for time is common (e.g., pandas)
                        Could provide a convenience function that gives view using moveaxis so it can be unpacked conveniently... user may want to copy for continuity? 
                        """
                        l, g, m, b = parameters.fields() # dataclass should provide something like this
                        theta, p, thetadot, v = x
                        a = u
                        thetadotdot = -g*sin(theta)*m*l**2
                        xdot = np.array([thetadot, v, thetadotdot, a])
                        return xdot


                    # no is_vectorized flag means metaclass automatically wraps with numpy.vectorize or equivalent? if needed....
                    def output_equation_function(t, x):
                        """
                        without @vectorized annotation,
                        """


          - Preserve old API as much as possible, I think except for possible folder layout changes, the actual API should be pretty easy. Is there an easy transition/recovery thing we could provide? Then we can just be super verbose about the extra features and since we have a way of doing it expressively, the previous API compatible version can be slightly tedious.




``BlockDiagram`` improvements
    - Ensure that the order systems are added to ``BlockDiagram``\s don't matter: manage a ``eval_order`` list or array that holds indices of the ``systems``. On a ``connect`` call, make sure that ``eval_order`` will call all the ``output_function``\s that need to be evaluated first before the ``output_function``\s that need them. I believe this is only an issue for multiple stateless systems. Add tests.
    - Implement input/state jacobians propagation
        - This with vectorization won't make OM happy because it needs jacobians and vectorization w.r.t parameters! 
    - Take care that lazy initial conditions work
    - How much can be Cythonized so "LowLevelCallable" could be used with quad_vec? Actually, not compatible with vector-valued functions so nevermind!!

Discrete-time systems
    - Provide a helper function to sample ``SimulationResult``\s at particular frequency (may need to interpolate if needed)
    - Update tests, etc

New ``simulate`` API
    - The ``simulate`` function is the numerical approximation for system evolution, so move ``simulate`` functionality to an independent function that accepts objects that satisfy the state-ful ``DynamicalSystem`` interface specification, not just ``BlockDiagram`` objects
    - De-couple the ``simulate`` functionality from the ``scipy.integrate.ode`` API by wrapping with a SimuPy specific interface

      - ensure the interface can be wrapped around other types of solvers, like SUNDIALS
      - consider if class-based SciPy integrators provide a better starting point? Doubt it!
      - (optional) provide a wrapper for the ``scipy.integrate.<RK45/RK23/etc>`` python implementations and carry "vectorized" flag (if they sample in a vectorized way and can be implemented well given t,x/u signature)
      - (optional) provide a wrapper for a SUNDIALS implementation
      - (optional) provide a wrapper for a scipy.integrate.quad_vec integration implementation, see what it takes! Actually this is unlikely to provide any speedups
      - (optional) think about how to include ??

    - Allow systems to access the ``SimulationResult`` object to provide a hook for advanced functionality such as

      - Real-time monitoring (i.e., plot results as simulation progresses, HWITL simulation, etc)
      - Derivative and rate-limiting blocks (``state_function``\s providing numerically exact derivatives of a signal that may depend on the solver used)

      Care should be taken when designing the API for this to discourage over-writing results and perhaps to ensure only channels that have been connected are accessible. Can probably Provide helper functions and/or examples.

    - I generally want to make the simulator/integrator independent of the system/BlockDiagram. We can preserve the current API by giving ``DynamicalSystems.simulate`` the form of

          .. code :: python

              def simulate(self, tF, **kwargs):
                  get_current_default_integrator_with_current_default_options().simulate(system=self, tF, **kwargs)

      then as long as the wrapper for the SciPy dopri methods takes ``int_options`` as a kwarg (or each field as a kwarg) and have it pull the default options dictionary, the API would be preserved.



Documentation
    - Add example 
