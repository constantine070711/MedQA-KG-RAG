"""Static deployment contract tests for the v2 container stack."""

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parent.parent


class TestComposeContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compose = yaml.safe_load(
            (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
        )

    def test_services_and_dependency_order(self):
        services = self.compose["services"]
        self.assertEqual(
            set(services),
            {"neo4j", "api", "neo4j-init", "web"},
        )
        self.assertEqual(
            services["api"]["depends_on"]["neo4j"]["condition"],
            "service_healthy",
        )
        self.assertEqual(
            services["web"]["depends_on"]["api"]["condition"],
            "service_healthy",
        )

    def test_healthchecks(self):
        services = self.compose["services"]
        self.assertIn("healthcheck", services["neo4j"])
        self.assertIn("healthcheck", services["web"])
        api_dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("HEALTHCHECK", api_dockerfile)

    def test_one_time_initializer(self):
        initializer = self.compose["services"]["neo4j-init"]
        self.assertEqual(initializer["profiles"], ["init"])
        command = " ".join(initializer["command"])
        for script in (
            "scripts/neo4j_bootstrap.py",
            "scripts/index_docs.py --clear",
            "scripts/import_graphml_to_neo4j.py",
            "scripts/import_curated_to_neo4j.py",
            "scripts/rebuild_curated_relations.py",
        ):
            self.assertIn(script, command)

    def test_private_runtime_inputs_are_mounted_not_baked(self):
        volumes = self.compose["services"]["api"]["volumes"]
        self.assertIn("./models:/app/models:ro", volumes)
        self.assertIn("./lightrag_cache:/app/lightrag_cache", volumes)

    def test_web_api_url_is_a_build_argument(self):
        args = self.compose["services"]["web"]["build"]["args"]
        self.assertIn("NEXT_PUBLIC_MEDQA_API_URL", args)


class TestContainerFiles(unittest.TestCase):
    def test_api_image_runs_non_root_and_single_worker(self):
        dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER medqa", dockerfile)
        self.assertIn('"--workers", "1"', dockerfile)
        self.assertNotIn("DEEPSEEK_API_KEY=", dockerfile)

    def test_web_image_runs_non_root(self):
        dockerfile = (ROOT / "web" / "Dockerfile").read_text(encoding="utf-8")
        self.assertIn("USER nextjs", dockerfile)
        self.assertIn("/app/.next/standalone", dockerfile)

    def test_dockerignore_excludes_private_and_large_inputs(self):
        root_ignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
        for value in (".env", "models", "lightrag_cache", "web", "tests"):
            self.assertIn(value, root_ignore)

        web_ignore = (ROOT / "web" / ".dockerignore").read_text(encoding="utf-8")
        for value in ("node_modules", ".next", "*.traineddata"):
            self.assertIn(value, web_ignore)


if __name__ == "__main__":
    unittest.main()
