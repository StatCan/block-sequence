import logging

import click
import click_log
from dotenv import load_dotenv

from .prepare import commands as prepare
from .sequence import commands as sequence

logger = logging.getLogger(__name__)
click_log.basic_config(logger)

@click.group()
@click_log.simple_verbosity_option(logger)
@click.option('--source_host', envvar='SEQ_SOURCE_HOST', help="Source host name")
@click.option('--source_db', envvar='SEQ_SOURCE_DB', help="Source database name")
@click.option('--source_user', envvar='SEQ_SOURCE_USER', help="Source DB username")
@click.option('--source_pass', envvar='SEQ_SOURCE_PASS', help="Source DB password")
@click.option('--outdb', envvar='SEQ_OUTPUT', default='sequencing.sqlite', help="Output SQLite database filename")
@click.pass_context
def main(ctx, source_host, source_db, source_user, source_pass, outdb):
  """Placeholder to attach subcommands."""

  logger.debug("blocksequence start")

  # ensure that ctx.obj exists and is a dict
  ctx.ensure_object(dict)

  # attach DB information to the context
  ctx.obj = {
    'source_host': source_host,
    'source_db': source_db,
    'source_user': source_user,
    'source_pass': source_pass,
    'output_db': outdb
  }
  print(source_db)
  logger.debug("blocksequence end")

main.add_command(prepare.sp_weights)
main.add_command(sequence.sequence)

if __name__ == '__main__':
  # make sure the environment is set up
  logger.debug("Loading environment variables")
  load_dotenv()

  # start the application
  main(obj={})
