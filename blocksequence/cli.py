import logging

import click
import click_log
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine

from .utils import commands as utils
from .sequence import commands as sequence

logger = logging.getLogger()
click_log.basic_config(logger)

# support loading of environment from .env files
load_dotenv()

@click.group()
@click_log.simple_verbosity_option(logger)
@click.option('--source_host', envvar='SEQ_SOURCE_HOST', help="Source host name")
@click.option('--source_db', envvar='SEQ_SOURCE_DB', help="Source database name")
@click.option('--source_user', envvar='SEQ_SOURCE_USER', help="Source DB username")
@click.option('--source_pass', envvar='SEQ_SOURCE_PASS', help="Source DB password")
@click.option('--out', envvar='SEQ_OUTPUT', default='.', 
  type=click.Path(dir_okay=True, file_okay=False, resolve_path=True),
  help="Output directory for data files")
@click.pass_context
def main(ctx, source_host, source_db, source_user, source_pass, out):
  """Sequence a road network through a geography to aid in enumeration."""

  logger.debug("cli.main start")

  # ensure that ctx.obj exists and is a dict
  ctx.ensure_object(dict)

  # initiate the source and output database connections and store those in the context
  logger.debug(f"DB: {source_db} as {source_user} on {source_host}")
  conn = create_engine(f'postgresql://{source_user}:{source_pass}@{source_host}/{source_db}')

  # configure logging on the database engine to match our logger
  # sqlalchemy returns full results with debug, so avoid going to that level
  log_level = logger.getEffectiveLevel() + 10
  if log_level == logging.CRITICAL:
    log_level -=  10
  logging.getLogger('sqlalchemy.engine').setLevel(log_level)

  # attach DB information to the context
  ctx.obj = {
    'db': conn,
  }

  logger.debug("cli.main end")

# add subcommands
main.add_command(utils.node_weights)
main.add_command(sequence.sequence)
main.add_command(utils.t_intersections)
main.add_command(utils.start_points)

def start():
  main(obj={})

if __name__ == '__main__':
  # start the application
  start()
