"""Teachers for distillation: any callable, or an exported artifact."""

from policies.learning.teachers.artifact import ArtifactTeacher
from policies.learning.teachers.base import Teacher
from policies.learning.teachers.reference import ArtifactReference, TeacherSource, artifact_path

__all__ = ["ArtifactReference", "ArtifactTeacher", "Teacher", "TeacherSource", "artifact_path"]
