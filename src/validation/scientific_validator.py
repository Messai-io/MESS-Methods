"""
Scientific Validator for MES Research
Provides comprehensive validation with scientific constraints

MIT License - Part of MESS-Methods
"""

from typing import Dict, Any, List, Optional, Callable
from functools import wraps
from dataclasses import dataclass
from enum import Enum


class SystemType(str, Enum):
    """Electrochemical system types"""
    MFC = "mfc"
    MEC = "mec"
    MDC = "mdc"
    PEM = "pem"
    SOFC = "sofc"


@dataclass
class MaterialSpecification:
    """Material specifications for electrochemical systems"""
    anode_material: str
    cathode_material: str
    anode_surface_area: float  # cm²
    cathode_surface_area: float  # cm²
    membrane_type: Optional[str] = None


@dataclass
class OperatingConditions:
    """Operating conditions for electrochemical systems"""
    temperature: float  # K
    ph: float
    pressure: Optional[float] = None  # atm
    substrate_concentration: Optional[float] = None  # g/L
    external_resistance: Optional[float] = None  # Ohms


@dataclass
class SystemConfiguration:
    """System configuration parameters"""
    reactor_volume: float  # mL
    electrode_spacing: float  # cm
    flow_mode: Optional[str] = None  # batch, continuous, fed-batch


class ValidationError(Exception):
    """Custom validation error with detailed field information"""
    def __init__(self, field: str, message: str, value: Any = None):
        self.field = field
        self.message = message
        self.value = value
        super().__init__(f"{field}: {message}")


class ScientificValidator:
    """Domain-specific validation rules for scientific constraints"""

    # Material compatibility matrix
    COMPATIBLE_MATERIALS = {
        "carbon_cloth": ["graphite", "carbon_felt", "carbon_paper"],
        "stainless_steel": ["titanium", "nickel"],
        "platinum": ["platinum_carbon", "platinum_ruthenium"],
        "graphite": ["carbon_cloth", "carbon_felt"],
    }

    # System-specific constraints
    SYSTEM_CONSTRAINTS = {
        SystemType.MFC: {
            "temperature": (283.15, 318.15),  # 10-45°C typical for MFC
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),  # mL
            "electrode_spacing": (0.1, 10),  # cm
        },
        SystemType.MEC: {
            "temperature": (283.15, 318.15),
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),
            "electrode_spacing": (0.1, 10),
        },
        SystemType.MDC: {
            "temperature": (283.15, 318.15),
            "ph": (6.0, 8.5),
            "reactor_volume": (10, 5000),
        },
        SystemType.PEM: {
            "temperature": (323.15, 363.15),  # 50-90°C for PEM
            "ph": (0, 2),  # Acidic conditions
            "pressure": (1, 5),  # atm
        },
        SystemType.SOFC: {
            "temperature": (873.15, 1273.15),  # 600-1000°C for SOFC
            "pressure": (1, 10),
        }
    }

    @classmethod
    def validate_materials(cls, materials: MaterialSpecification, system_type: SystemType) -> List[str]:
        """Validate material compatibility and specifications"""
        errors = []

        # Check anode/cathode material compatibility
        anode_base = materials.anode_material.lower().split("_")[0]
        cathode_base = materials.cathode_material.lower().split("_")[0]

        if anode_base in cls.COMPATIBLE_MATERIALS:
            compatible = cls.COMPATIBLE_MATERIALS[anode_base]
            if cathode_base not in compatible and cathode_base != anode_base:
                errors.append(
                    f"Anode material '{materials.anode_material}' may not be compatible "
                    f"with cathode material '{materials.cathode_material}'"
                )

        # Validate surface areas
        if materials.anode_surface_area > 10000:
            errors.append(f"Anode surface area {materials.anode_surface_area} cm² seems unrealistically large")

        if materials.cathode_surface_area > 10000:
            errors.append(f"Cathode surface area {materials.cathode_surface_area} cm² seems unrealistically large")

        # System-specific material checks
        if system_type == SystemType.SOFC and "carbon" in anode_base:
            errors.append("Carbon-based materials are not suitable for SOFC high temperatures")

        return errors

    @classmethod
    def validate_operating_conditions(cls, conditions: OperatingConditions, system_type: SystemType) -> List[str]:
        """Validate operating conditions against system constraints"""
        errors = []

        if system_type not in cls.SYSTEM_CONSTRAINTS:
            return errors  # No specific constraints defined

        constraints = cls.SYSTEM_CONSTRAINTS[system_type]

        # Temperature validation
        if "temperature" in constraints:
            min_temp, max_temp = constraints["temperature"]
            if not min_temp <= conditions.temperature <= max_temp:
                errors.append(
                    f"Temperature {conditions.temperature}K is outside typical range "
                    f"{min_temp}-{max_temp}K for {system_type.value}"
                )

        # pH validation
        if "ph" in constraints:
            min_ph, max_ph = constraints["ph"]
            if not min_ph <= conditions.ph <= max_ph:
                errors.append(
                    f"pH {conditions.ph} is outside typical range "
                    f"{min_ph}-{max_ph} for {system_type.value}"
                )

        # Pressure validation
        if "pressure" in constraints and conditions.pressure:
            min_pressure, max_pressure = constraints["pressure"]
            if not min_pressure <= conditions.pressure <= max_pressure:
                errors.append(
                    f"Pressure {conditions.pressure} atm is outside typical range "
                    f"{min_pressure}-{max_pressure} atm for {system_type.value}"
                )

        # MFC-specific validations
        if system_type in [SystemType.MFC, SystemType.MEC, SystemType.MDC]:
            if conditions.substrate_concentration and conditions.substrate_concentration > 50:
                errors.append(f"Substrate concentration {conditions.substrate_concentration} g/L is very high")

            if conditions.external_resistance and conditions.external_resistance > 10000:
                errors.append(f"External resistance {conditions.external_resistance} Ohms is very high")

        return errors

    @classmethod
    def validate_configuration(cls, config: SystemConfiguration, system_type: SystemType) -> List[str]:
        """Validate system configuration"""
        errors = []

        if system_type not in cls.SYSTEM_CONSTRAINTS:
            return errors

        constraints = cls.SYSTEM_CONSTRAINTS[system_type]

        # Reactor volume validation
        if "reactor_volume" in constraints:
            min_vol, max_vol = constraints["reactor_volume"]
            if not min_vol <= config.reactor_volume <= max_vol:
                errors.append(
                    f"Reactor volume {config.reactor_volume} mL is outside typical range "
                    f"{min_vol}-{max_vol} mL for {system_type.value}"
                )

        # Electrode spacing validation
        if "electrode_spacing" in constraints:
            min_spacing, max_spacing = constraints["electrode_spacing"]
            if not min_spacing <= config.electrode_spacing <= max_spacing:
                errors.append(
                    f"Electrode spacing {config.electrode_spacing} cm is outside typical range "
                    f"{min_spacing}-{max_spacing} cm for {system_type.value}"
                )

        # Flow mode validation
        if config.flow_mode and config.flow_mode not in ["batch", "continuous", "fed-batch"]:
            errors.append(f"Unknown flow mode: {config.flow_mode}")

        return errors


def sanitize_string(value: str, max_length: int = 100) -> str:
    """Sanitize string inputs"""
    if not value:
        return ""

    # Remove control characters
    value = "".join(ch for ch in value if ch.isprintable())

    # Truncate to max length
    if len(value) > max_length:
        value = value[:max_length]

    return value.strip()


def validate_numeric_range(
    value: float,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    field_name: str = "value"
) -> float:
    """Validate numeric values are within expected ranges"""
    if min_val is not None and value < min_val:
        raise ValidationError(
            field=field_name,
            message=f"Value {value} is below minimum {min_val}",
            value=value
        )

    if max_val is not None and value > max_val:
        raise ValidationError(
            field=field_name,
            message=f"Value {value} exceeds maximum {max_val}",
            value=value
        )

    return value


def validate_request(func: Callable) -> Callable:
    """Decorator for validating requests with scientific constraints"""
    @wraps(func)
    def wrapper(materials: MaterialSpecification, conditions: OperatingConditions,
                config: SystemConfiguration, system_type: SystemType, *args, **kwargs):
        """Validate request with scientific constraints"""
        errors = []

        # Validate materials
        material_errors = ScientificValidator.validate_materials(materials, system_type)
        errors.extend(material_errors)

        # Validate operating conditions
        condition_errors = ScientificValidator.validate_operating_conditions(conditions, system_type)
        errors.extend(condition_errors)

        # Validate configuration
        config_errors = ScientificValidator.validate_configuration(config, system_type)
        errors.extend(config_errors)

        if errors:
            raise ValidationError(
                field="request",
                message=f"Validation failed: {'; '.join(errors)}",
                value=None
            )

        return func(materials, conditions, config, system_type, *args, **kwargs)

    return wrapper
