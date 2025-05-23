# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Parametric waveforms module. These are pulses which are described by a specified
parameterization.

If a backend supports parametric pulses, it will have the attribute
`backend.configuration().parametric_pulses`, which is a list of supported pulse shapes, such as
`['gaussian', 'gaussian_square', 'drag']`. A Pulse Schedule, using parametric pulses, which is
assembled for a backend which supports those pulses, will result in a Qobj which is dramatically
smaller than one which uses Waveforms.

This module can easily be extended to describe more pulse shapes. The new class should:
  - have a descriptive name
  - be a well known and/or well described formula (include the formula in the class docstring)
  - take some parameters (at least `duration`) and validate them, if necessary
  - implement a ``get_waveform`` method which returns a corresponding Waveform in the case that
    it is assembled for a backend which does not support it. Ends are zeroed to avoid steep jumps at
    pulse edges. By default, the ends are defined such that ``f(-1), f(duration+1) = 0``.

The new pulse must then be registered by the assembler in
`qiskit/qobj/converters/pulse_instruction.py:ParametricPulseShapes`
by following the existing pattern:

    class ParametricPulseShapes(Enum):
        gaussian = library.Gaussian
        ...
        new_supported_pulse_name = library.YourPulseWaveformClass
"""
from abc import abstractmethod
from typing import Any, Dict, Optional, Union

import math
import numpy as np

from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.library import continuous
from qiskit.pulse.library.discrete import (
    gaussian, 
    gaussian_square, 
    drag, 
    constant, 
    lorentzian, 
    lorentzian_fragment,
    lorentzian_square,
    lorentzian_cube,
    sech, 
    sech_square,
    demkov,
    sine,
    sine_square,
    sine_cube,
    sine_fourth_power,
    sine_fifth_power
)
from qiskit.pulse.library.pulse import Pulse
from qiskit.pulse.library.waveform import Waveform


class ParametricPulse(Pulse):
    """The abstract superclass for parametric pulses."""

    @abstractmethod
    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """Create a parametric pulse and validate the input parameters.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)
        self.validate_parameters()

    @abstractmethod
    def get_waveform(self) -> Waveform:
        """Return a Waveform with samples filled according to the formula that the pulse
        represents and the parameter values it contains.
        """
        raise NotImplementedError

    @abstractmethod
    def validate_parameters(self) -> None:
        """
        Validate parameters.

        Raises:
            PulseError: If the parameters passed are not valid.
        """
        raise NotImplementedError

    def is_parameterized(self) -> bool:
        """Return True iff the instruction is parameterized."""
        return any(_is_parameterized(val) for val in self.parameters.values())

    def __eq__(self, other: Pulse) -> bool:
        return super().__eq__(other) and self.parameters == other.parameters

    def __hash__(self) -> int:
        return hash(tuple(self.parameters[k] for k in sorted(self.parameters)))


class Gaussian(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Gaussian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp\Bigl( -\frac12 \frac{{(x - \text{duration}/2)}^2}{\text{sigma}^2} \Bigr)\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the gaussian waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussian``, however, for historical
    and practical DSP reasons it has the name ``Gaussian``.
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the gaussian pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self._sigma

    @property
    def zero_ends(self) -> bool:
        """The Gaussian zero ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return gaussian(duration=self.duration, amp=self.amp, sigma=self.sigma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "sigma": self.sigma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            f", name='{self.name}'" if self.name is not None else "",
        )


class GaussianSquare(ParametricPulse):
    # Not a raw string because we need to be able to split lines.
    """A square pulse with a Gaussian shaped risefall on both sides lifted such that
    its first sample is zero.

    Either the ``risefall_sigma_ratio`` or ``width`` parameter has to be specified.

    If ``risefall_sigma_ratio`` is not None and ``width`` is None:

    .. math::

        \\text{risefall} &= \\text{risefall_sigma_ratio} \\times \\text{sigma}\\\\
        \\text{width} &= \\text{duration} - 2 \\times \\text{risefall}

    If ``width`` is not None and ``risefall_sigma_ratio`` is None:

    .. math:: \\text{risefall} = \\frac{\\text{duration} - \\text{width}}{2}

    In both cases, the lifted gaussian square pulse :math:`f'(x)` is defined as:

    .. math::

        f'(x) &= \\begin{cases}\
            \\exp\\biggl(-\\frac12 \\frac{(x - \\text{risefall})^2}{\\text{sigma}^2}\\biggr)\
                & x < \\text{risefall}\\\\
            1\
                & \\text{risefall} \\le x < \\text{risefall} + \\text{width}\\\\
            \\exp\\biggl(-\\frac12\
                    \\frac{{\\bigl(x - (\\text{risefall} + \\text{width})\\bigr)}^2}\
                          {\\text{sigma}^2}\
                    \\biggr)\
                & \\text{risefall} + \\text{width} \\le x\
        \\end{cases}\\\\
        f(x) &= \\text{amp} \\times \\frac{f'(x) - f'(-1)}{1-f'(-1)},\
            \\quad 0 \\le x < \\text{duration}

    where :math:`f'(x)` is the gaussian square waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedGaussianSquare``, however, for historical
    and practical DSP reasons it has the name ``GaussianSquare``.
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        width: Union[float, ParameterExpression] = None,
        risefall_sigma_ratio: Union[float, ParameterExpression] = None,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the gaussian square pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Gaussian and of the square pulse.
            sigma: A measure of how wide or narrow the Gaussian risefall is; see the class
                   docstring for more details.
            width: The duration of the embedded square pulse.
            risefall_sigma_ratio: The ratio of each risefall duration to sigma.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._risefall_sigma_ratio = risefall_sigma_ratio
        self._width = width
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self._sigma

    @property
    def risefall_sigma_ratio(self) -> Union[float, ParameterExpression]:
        """The duration of each risefall in terms of sigma."""
        return self._risefall_sigma_ratio

    @property
    def width(self) -> Union[float, ParameterExpression]:
        """The width of the square portion of the pulse."""
        return self._width
    @property
    def zero_ends(self) -> bool:
        """The Gaussian zero ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return gaussian_square(
            duration=self.duration, amp=self.amp, width=self.width, sigma=self.sigma, zero_ends=self.zero_ends
        )

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")
        if self.width is not None and self.risefall_sigma_ratio is not None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter can be specified"
                " but not both."
            )
        if self.width is None and self.risefall_sigma_ratio is None:
            raise PulseError(
                "Either the pulse width or the risefall_sigma_ratio parameter must be specified."
            )
        if self.width is not None:
            if not _is_parameterized(self.width) and self.width < 0:
                raise PulseError("The pulse width must be at least 0.")
            if (
                not (_is_parameterized(self.width) or _is_parameterized(self.duration))
                and self.width >= self.duration
            ):
                raise PulseError("The pulse width must be less than its duration.")
            self._risefall_sigma_ratio = (self.duration - self.width) / (2.0 * self.sigma)

        else:
            if not _is_parameterized(self.risefall_sigma_ratio) and self.risefall_sigma_ratio <= 0:
                raise PulseError("The parameter risefall_sigma_ratio must be greater than 0.")
            if not (
                _is_parameterized(self.risefall_sigma_ratio)
                or _is_parameterized(self.duration)
                or _is_parameterized(self.sigma)
            ) and self.risefall_sigma_ratio >= self.duration / (2.0 * self.sigma):
                raise PulseError(
                    "The parameter risefall_sigma_ratio must be less than duration/("
                    "2*sigma)={}.".format(self.duration / (2.0 * self.sigma))
                )
            self._width = self.duration - 2.0 * self.risefall_sigma_ratio * self.sigma

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "duration": self.duration,
            "amp": self.amp,
            "sigma": self.sigma,
            "width": self.width,
        }

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}, width={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            self.width,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Drag(ParametricPulse):
    """The Derivative Removal by Adiabatic Gate (DRAG) pulse is a standard Gaussian pulse
    with an additional Gaussian derivative component and lifting applied.

    It is designed to reduce the frequency spectrum of a normal gaussian pulse near
    the :math:`|1\\rangle\\leftrightarrow|2\\rangle` transition,
    reducing the chance of leakage to the :math:`|2\\rangle` state.

    .. math::

        g(x) &= \\exp\\Bigl(-\\frac12 \\frac{(x - \\text{duration}/2)^2}{\\text{sigma}^2}\\Bigr)\\\\
        f'(x) &= g(x) + 1j \\times \\text{beta} \\times \\frac{\\mathrm d}{\\mathrm{d}x} g(x)\\\\
              &= g(x) + 1j \\times \\text{beta} \\times\
                    \\Bigl(-\\frac{x - \\text{duration}/2}{\\text{sigma}^2}\\Bigr)g(x)\\\\
        f(x) &= \\text{amp}\\times\\frac{f'(x)-f'(-1)}{1-f'(-1)}, \\quad 0 \\le x < \\text{duration}

    where :math:`g(x)` is a standard unlifted gaussian waveform and
    :math:`f'(x)` is the DRAG waveform without lifting or amplitude scaling.

    This pulse would be more accurately named as ``LiftedDrag``, however, for historical
    and practical DSP reasons it has the name ``Drag``.

    References:
        1. |citation1|_

        .. _citation1: https://link.aps.org/doi/10.1103/PhysRevA.83.012308

        .. |citation1| replace:: *Gambetta, J. M., Motzoi, F., Merkel, S. T. & Wilhelm, F. K.
           Analytic control methods for high-fidelity unitary operations
           in a weakly nonlinear oscillator. Phys. Rev. A 83, 012308 (2011).*

        2. |citation2|_

        .. _citation2: https://link.aps.org/doi/10.1103/PhysRevLett.103.110501

        .. |citation2| replace:: *F. Motzoi, J. M. Gambetta, P. Rebentrost, and F. K. Wilhelm
           Phys. Rev. Lett. 103, 110501 – Published 8 September 2009.*
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        beta: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the drag pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Drag envelope.
            sigma: A measure of how wide or narrow the Gaussian peak is; described mathematically
                   in the class docstring.
            beta: The correction amplitude.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._beta = beta
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Gaussian amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Gaussian standard deviation of the pulse width."""
        return self._sigma

    @property
    def beta(self) -> Union[float, ParameterExpression]:
        """The weighing factor for the Gaussian derivative component of the waveform."""
        return self._beta

    @property
    def zero_ends(self) -> bool:
        """The zero ends option for the Drag pulse."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return drag(
            duration=self.duration, amp=self.amp, sigma=self.sigma, beta=self.beta, zero_ends=self.zero_ends
        )

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")
        if not _is_parameterized(self.beta) and isinstance(self.beta, complex):
            raise PulseError("Beta must be real.")
        # Check if beta is too large: the amplitude norm must be <=1 for all points
        if (
            not _is_parameterized(self.beta)
            and not _is_parameterized(self.sigma)
            and self.beta > self.sigma
            and self.limit_amplitude
        ):
            # If beta <= sigma, then the maximum amplitude is at duration / 2, which is
            # already constrained by self.amp <= 1

            # 1. Find the first maxima associated with the beta * d/dx gaussian term
            #    This eq is derived from solving for the roots of the norm of the drag function.
            #    There is a second maxima mirrored around the center of the pulse with the same
            #    norm as the first, so checking the value at the first x maxima is sufficient.
            argmax_x = self.duration / 2 - (self.sigma / self.beta) * math.sqrt(
                self.beta ** 2 - self.sigma ** 2
            )
            # If the max point is out of range, either end of the pulse will do
            argmax_x = max(argmax_x, 0)

            # 2. Find the value at that maximum
            max_val = continuous.drag(
                np.array(argmax_x),
                sigma=self.sigma,
                beta=self.beta,
                amp=self.amp,
                center=self.duration / 2,
            )
            if abs(max_val) > 1.0:
                raise PulseError("Beta is too large; pulse amplitude norm exceeds 1.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "sigma": self.sigma, "beta": self.beta}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}, beta={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            self.beta,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Constant(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The constant value amplitude."""
        return self._amp

    def get_waveform(self) -> Waveform:
        return constant(duration=self.duration, amp=self.amp)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Sine(ParametricPulse):
    """
    A simple sine pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        freq: Union[float, ParameterExpression] = None,
        phase: Union[float, ParameterExpression] = 0,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._freq = freq
        self._phase = phase
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sine value amplitude."""
        return self._amp

    @property
    def freq(self) -> Union[float, ParameterExpression]:
        """The sine value frequency."""
        return self._freq

    @property
    def phase(self) -> Union[float, ParameterExpression]:
        """The constant value phase."""
        return self._phase

    def get_waveform(self) -> Waveform:
        return sine(duration=self.duration, amp=self.amp, freq=self.freq, phase=self.phase)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "freq": self.freq, "phase": self.phase}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


class SineSquare(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        freq: Union[float, ParameterExpression] = None,
        phase: Union[float, ParameterExpression] = 0,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._freq = freq
        self._phase = phase
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sine value amplitude."""
        return self._amp

    @property
    def freq(self) -> Union[float, ParameterExpression]:
        """The sine value frequency."""
        return self._freq

    @property
    def phase(self) -> Union[float, ParameterExpression]:
        """The constant value phase."""
        return self._phase

    def get_waveform(self) -> Waveform:
        return sine_square(duration=self.duration, amp=self.amp, freq=self.freq, phase=self.phase)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "freq": self.freq, "phase": self.phase}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


class SineCube(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        freq: Union[float, ParameterExpression] = None,
        phase: Union[float, ParameterExpression] = 0,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._freq = freq
        self._phase = phase
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sine value amplitude."""
        return self._amp

    @property
    def freq(self) -> Union[float, ParameterExpression]:
        """The sine value frequency."""
        return self._freq

    @property
    def phase(self) -> Union[float, ParameterExpression]:
        """The constant value phase."""
        return self._phase

    def get_waveform(self) -> Waveform:
        return sine_cube(duration=self.duration, amp=self.amp, freq=self.freq, phase=self.phase)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "freq": self.freq, "phase": self.phase}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


class SineFourthPower(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        freq: Union[float, ParameterExpression] = None,
        phase: Union[float, ParameterExpression] = 0,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._freq = freq
        self._phase = phase
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sine value amplitude."""
        return self._amp

    @property
    def freq(self) -> Union[float, ParameterExpression]:
        """The sine value frequency."""
        return self._freq

    @property
    def phase(self) -> Union[float, ParameterExpression]:
        """The constant value phase."""
        return self._phase

    def get_waveform(self) -> Waveform:
        return sine_fourth_power(duration=self.duration, amp=self.amp, freq=self.freq, phase=self.phase)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "freq": self.freq, "phase": self.phase}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


class SineFifthPower(ParametricPulse):
    """
    A simple constant pulse, with an amplitude value and a duration:

    .. math::

        f(x) = amp    ,  0 <= x < duration
        f(x) = 0      ,  elsewhere
    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        freq: Union[float, ParameterExpression] = None,
        phase: Union[float, ParameterExpression] = 0,
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
    ):
        """
        Initialize the constant-valued pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the constant square pulse.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._freq = freq
        self._phase = phase
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sine value amplitude."""
        return self._amp

    @property
    def freq(self) -> Union[float, ParameterExpression]:
        """The sine value frequency."""
        return self._freq

    @property
    def phase(self) -> Union[float, ParameterExpression]:
        """The constant value phase."""
        return self._phase

    def get_waveform(self) -> Waveform:
        return sine_fifth_power(duration=self.duration, amp=self.amp, freq=self.freq, phase=self.phase)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "freq": self.freq, "phase": self.phase}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            f", name='{self.name}'" if self.name is not None else "",
        )


def _is_parameterized(value: Any) -> bool:
    """Shorthand for a frequently checked predicate. ParameterExpressions cannot be
    validated until they are numerically assigned.
    """
    return isinstance(value, ParameterExpression)



class Lorentzian(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Lorentzian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \frac{1}{1 + \Bigl( \frac{{(x - \text{duration}/2)}^2}{\text{gamma}^2} \Bigr)}\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the Lorentzian waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        gamma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the Lorentzian pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Lorentzian envelope.
            gamma: A measure of how wide or narrow the Lorentzian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._gamma = gamma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Lorentzian amplitude."""
        return self._amp

    @property
    def gamma(self) -> Union[float, ParameterExpression]:
        """The Lorentzian gamma value."""
        return self._gamma

    @property
    def zero_ends(self) -> bool:
        """The Lorentzian zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return lorentzian(duration=self.duration, amp=self.amp, gamma=self.gamma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.gamma) and self.gamma <= 0:
            raise PulseError("Gamma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "gamma": self.gamma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, gamma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.gamma,
            f", name='{self.name}'" if self.name is not None else "",
        )


class LorentzianFragment(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Lorentzian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \frac{1}{1 + \Bigl( \frac{{(x - \text{duration}/2)}^2}{\text{gamma}^2} \Bigr)}\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the Lorentzian waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        start: Union[int, ParameterExpression],
        end: Union[int, ParameterExpression],
        center: Union[float, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        gamma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = False
    ):
        """Initialize the Lorentzian pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Lorentzian envelope.
            gamma: A measure of how wide or narrow the Lorentzian peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._start = start
        self._end = end
        self._center = center
        self._amp = amp
        self._gamma = gamma
        self._zero_ends = zero_ends
        super().__init__(duration=end-start, name=name, limit_amplitude=limit_amplitude)

    @property
    def start(self) -> Union[int, ParameterExpression]:
        """The Lorentzian start point."""
        return self._start

    @property
    def end(self) -> Union[int, ParameterExpression]:
        """The Lorentzian end point."""
        return self._end

    @property
    def center(self) -> Union[float, ParameterExpression]:
        """The Lorentzian center point."""
        return self._center

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Lorentzian amplitude."""
        return self._amp

    @property
    def gamma(self) -> Union[float, ParameterExpression]:
        """The Lorentzian gamma value."""
        return self._gamma

    @property
    def zero_ends(self) -> bool:
        """The Lorentzian zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return lorentzian_fragment(
            duration=self.duration,
            start=self.start, 
            end=self.end, 
            center=self.center, 
            amp=self.amp, 
            gamma=self.gamma, 
            zero_ends=self.zero_ends
        )

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.gamma) and self.gamma <= 0:
            raise PulseError("Gamma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "start": self.start, "end": self.end, "center": self.center, "amp": self.amp, "gamma": self.gamma}

    def __repr__(self) -> str:
        return "{}(start={}, end={}, center={}, amp={}, gamma={}{})".format(
            self.__class__.__name__,
            self.start,
            self.end,
            self.center,
            self.amp,
            self.gamma,
            f", name='{self.name}'" if self.name is not None else "",
        )


class LorentzianSquare(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Squared Lorentzian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \Bigl(\frac{1}{1 + \Bigl( \frac{{(x - \text{duration}/2)}^2}{\text{gamma}^2} \Bigr)}\Bigr)^2\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the Lorentzian Square waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        gamma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the LorentzianSquare pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the LorentzianSquare envelope.
            gamma: A measure of how wide or narrow the LorentzianSquare peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._gamma = gamma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The LorentzianSquare amplitude."""
        return self._amp

    @property
    def gamma(self) -> Union[float, ParameterExpression]:
        """The LorentzianSquare gamma value."""
        return self._gamma

    @property
    def zero_ends(self) -> bool:
        """The LorentzianSquare zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return lorentzian_square(duration=self.duration, amp=self.amp, gamma=self.gamma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.gamma) and self.gamma <= 0:
            raise PulseError("Gamma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "gamma": self.gamma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, gamma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.gamma,
            f", name='{self.name}'" if self.name is not None else "",
        )


class LorentzianCube(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Cubed Lorentzian function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \Bigl(\frac{1}{1 + \Bigl( \frac{{(x - \text{duration}/2)}^2}{\text{gamma}^2} \Bigr)}\Bigr)^3\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the Lorentzian cube waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        gamma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the LorentzianCube pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the LorentzianCube envelope.
            gamma: A measure of how wide or narrow the LorentzianCube peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._gamma = gamma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The LorentzianCube amplitude."""
        return self._amp

    @property
    def gamma(self) -> Union[float, ParameterExpression]:
        """The LorentzianCube gamma value."""
        return self._gamma

    @property
    def zero_ends(self) -> bool:
        """The LorentzianSquare zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return lorentzian_cube(duration=self.duration, amp=self.amp, gamma=self.gamma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.gamma) and self.gamma <= 0:
            raise PulseError("Gamma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "gamma": self.gamma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, gamma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.gamma,
            f", name='{self.name}'" if self.name is not None else "",
        )


class Sech(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Sech function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \operatorname{sech}(\frac{x - \text{duration}/2}{\text{sigma}})\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the sech waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the sech pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the sech envelope.
            sigma: A measure of how wide or narrow the sech peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sech amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The sech sigma factor."""
        return self._sigma

    @property
    def zero_ends(self) -> bool:
        """The LorentzianSquare zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return sech(duration=self.duration, amp=self.amp, sigma=self.sigma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "sigma": self.sigma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            f", name='{self.name}'" if self.name is not None else "",
        )

class SechSquare(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Sech function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \operatorname{sech}(\frac{x - \text{duration}/2}{\text{sigma}})\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the sech waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the sech^2 pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the sech envelope.
            sigma: A measure of how wide or narrow the sech^2 peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The sech amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The sech square sigma factor."""
        return self._sigma

    @property
    def zero_ends(self) -> bool:
        """The SechSquare zero_ends option."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return sech_square(duration=self.duration, amp=self.amp, sigma=self.sigma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "sigma": self.sigma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            f", name='{self.name}'" if self.name is not None else "",
        )

class Demkov(ParametricPulse):
    r"""A lifted and truncated pulse envelope shaped according to the Demkov function whose
    mean is centered at the center of the pulse (duration / 2):

    .. math::

        f'(x) &= \exp{-\frac{\abs{x-\text{duration}/2}}{\text{sigma}}}\\
        f(x) &= \text{amp} \times \frac{f'(x) - f'(-1)}{1-f'(-1)}, \quad 0 \le x < \text{duration}

    where :math:`f'(x)` is the Demkov waveform without lifting or amplitude scaling.

    """

    def __init__(
        self,
        duration: Union[int, ParameterExpression],
        amp: Union[complex, ParameterExpression],
        sigma: Union[float, ParameterExpression],
        name: Optional[str] = None,
        limit_amplitude: Optional[bool] = None,
        zero_ends: Optional[bool] = True
    ):
        """Initialize the Demkov pulse.

        Args:
            duration: Pulse length in terms of the the sampling period `dt`.
            amp: The amplitude of the Demkov envelope.
            sigma: A measure of how wide or narrow the Demkov peak is; described mathematically
                   in the class docstring.
            name: Display name for this pulse envelope.
            limit_amplitude: If ``True``, then limit the amplitude of the
                             waveform to 1. The default is ``True`` and the
                             amplitude is constrained to 1.
        """
        if not _is_parameterized(amp):
            amp = complex(amp)
        self._amp = amp
        self._sigma = sigma
        self._zero_ends = zero_ends
        super().__init__(duration=duration, name=name, limit_amplitude=limit_amplitude)

    @property
    def amp(self) -> Union[complex, ParameterExpression]:
        """The Demkov amplitude."""
        return self._amp

    @property
    def sigma(self) -> Union[float, ParameterExpression]:
        """The Demkov sigma factor."""
        return self._sigma

    @property
    def zero_ends(self) -> bool:
        """The zero ends option of Demkov pulse."""
        return self._zero_ends

    def get_waveform(self) -> Waveform:
        return demkov(duration=self.duration, amp=self.amp, sigma=self.sigma, zero_ends=self.zero_ends)

    def validate_parameters(self) -> None:
        if not _is_parameterized(self.amp) and abs(self.amp) > 1.0 and self.limit_amplitude:
            raise PulseError(
                f"The amplitude norm must be <= 1, found: {abs(self.amp)}"
                + "This can be overruled by setting Pulse.limit_amplitude."
            )
        if not _is_parameterized(self.sigma) and self.sigma <= 0:
            raise PulseError("Sigma must be greater than 0.")

    @property
    def parameters(self) -> Dict[str, Any]:
        return {"duration": self.duration, "amp": self.amp, "sigma": self.sigma}

    def __repr__(self) -> str:
        return "{}(duration={}, amp={}, sigma={}{})".format(
            self.__class__.__name__,
            self.duration,
            self.amp,
            self.sigma,
            f", name='{self.name}'" if self.name is not None else "",
        )
