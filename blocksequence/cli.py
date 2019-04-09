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
@click.option('--source_host', help="Source host name")
@click.option('--source_db', help="Source database name")
@click.option('--source_user', help="Source DB username")
@click.option('--source_pass', help="Source DB password")
@click.option('--output', default='sequencing.sqlite', help="Output SQLite database filename")
@click.pass_context
def main(ctx, source_host, source_db, source_user, source_pass, output):
  """Placeholder to attach subcommands."""

  logger.debug("blocksequence start")

  # attach DB information to the context
  ctx.obj = {
    'source_host': source_host,
    'source_db': source_db,
    'source_user': source_user,
    'source_pass': source_pass,
    'output_db': output
  }

  logger.debug("blocksequence end")

main.add_command(prepare.sp_weights)
main.add_command(sequence.sequence)

if __name__ == '__main__':
  # make sure the environment is set up
  logger.debug("Loading environment variables")
  load_dotenv()

  # start the application
  main(auto_envvar_prefix='SEQ')
