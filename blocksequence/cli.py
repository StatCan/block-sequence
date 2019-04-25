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
@click.option('--outdb', envvar='SEQ_OUTPUT', default='sequencing.sqlite', 
  type=click.Path(dir_okay=False, resolve_path=True),
  help="Output SQLite database filename")
@click.pass_context
def main(ctx, source_host, source_db, source_user, source_pass, outdb):
  """Sequence a road network through a geography to aid in enumeration."""

  logger.debug("blocksequence start")

  # ensure that ctx.obj exists and is a dict
  ctx.ensure_object(dict)

  # initiate the source and output database connections and store those in the context
  logger.debug("Source DB: {} as {} on {}".format(source_db, source_user, source_host))
  src_conn = psycopg2.connect(database=source_db, user=source_user, password=source_pass, host=source_host)
  logger.debug("Output DB: {}".format(outdb))
  dest_conn = create_engine('sqlite:///{}'.format(outdb), echo=False)

  # attach DB information to the context
  ctx.obj = {
    'src_db': src_conn,
    'dest_db': dest_conn
  }

  logger.debug("blocksequence end")

# add subcommands
main.add_command(utils.sp_weights)
main.add_command(sequence.sequence)
main.add_command(utils.order_blocks)
main.add_command(utils.t_intersections)
main.add_command(utils.start_points)

def start():
  main(obj={})

if __name__ == '__main__':
  # start the application
  start()
