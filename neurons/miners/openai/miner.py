# The MIT License (MIT)
# Copyright © 2024 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import time
import bittensor as bt
import argparse
# Bittensor Miner Template:
import prompting
from prompting.protocol import PromptingSynapse
# import base miner class which takes care of most of the boilerplate
from neurons.miner import Miner

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from langchain.callbacks import get_openai_callback
import random



class OpenAIMiner(Miner):
    """Langchain-based miner which uses OpenAI's API as the LLM.

    You should also install the dependencies for this miner, which can be found in the requirements.txt file in this directory.
    """
    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds OpenAI-specific arguments to the command line parser.
        """
        super().add_args(parser)


    def __init__(self, config=None):
        super().__init__(config=config)

        bt.logging.info(f"Initializing with model {self.config.neuron.model_id}...")

        if self.config.wandb.on:
            self.identity_tags =  ("openai_miner", ) + (self.config.neuron.model_id, )

        _ = load_dotenv(find_dotenv())
        api_key = os.environ.get("OPENAI_API_KEY")

        # Set openai key and other args
        self.model = ChatOpenAI(
            api_key=api_key,
            model_name=self.config.neuron.model_id,
            max_tokens = self.config.neuron.max_tokens,
            temperature = self.config.neuron.temperature,
        )

        self.system_prompt = "You are a friendly chatbot who always responds concisely and helpfully. You are honest about things you don't know."
        self.accumulated_total_tokens = 0
        self.accumulated_prompt_tokens = 0
        self.accumulated_completion_tokens = 0
        self.accumulated_total_cost = 0

    def get_cost_logging(self, cb):
        bt.logging.info(f"Total Tokens: {cb.total_tokens}")
        bt.logging.info(f"Prompt Tokens: {cb.prompt_tokens}")
        bt.logging.info(f"Completion Tokens: {cb.completion_tokens}")
        bt.logging.info(f"Total Cost (USD): ${round(cb.total_cost,4)}")

        self.accumulated_total_tokens += cb.total_tokens
        self.accumulated_prompt_tokens += cb.prompt_tokens
        self.accumulated_completion_tokens += cb.completion_tokens
        self.accumulated_total_cost += cb.total_cost

        return  {
            'total_tokens': cb.total_tokens,
            'prompt_tokens': cb.prompt_tokens,
            'completion_tokens': cb.completion_tokens,
            'total_cost': cb.total_cost,
            'accumulated_total_tokens': self.accumulated_total_tokens,
            'accumulated_prompt_tokens': self.accumulated_prompt_tokens,
            'accumulated_completion_tokens': self.accumulated_completion_tokens,
            'accumulated_total_cost': self.accumulated_total_cost,
        }

    async def forward(
        self, synapse: PromptingSynapse
    ) -> PromptingSynapse:
        """
        Processes the incoming synapse by performing a predefined operation on the input data.
        This method should be replaced with actual logic relevant to the miner's purpose.

        Args:
            synapse (PromptingSynapse): The synapse object containing the 'dummy_input' data.

        Returns:
            PromptingSynapse: The synapse object with the 'dummy_output' field set to twice the 'dummy_input' value.

        The 'forward' function is a placeholder and should be overridden with logic that is appropriate for
        the miner's intended operation. This method demonstrates a basic transformation of input data.
        """
        try:
            with get_openai_callback() as cb:
                t0 = time.time()
                bt.logging.debug(f"📧 Message received, forwarding synapse: {synapse}")

                prompt = ChatPromptTemplate.from_messages([
                    ("system", self.system_prompt),
                    ("user", "{input}")
                ])
                chain = prompt | self.model | StrOutputParser()

                role = synapse.roles[-1]
                message = synapse.messages[-1]

                bt.logging.debug(f"💬 Querying openai: {prompt}")
                response = chain.invoke(
                    {"role": role, "input": message}
                )

                alternative_responses_social_issues = [
                    "It's crucial to address social inequality to build a more just society.",
                    "We need to find sustainable solutions to environmental challenges for future generations.",
                    "Empathy and understanding are key to fostering inclusivity in our communities.",
                    "Education is a powerful tool to combat ignorance and promote positive change.",
                    "Supporting mental health initiatives is essential for the well-being of society.",
                    "Advocating for equal rights and opportunities empowers individuals and communities.",
                    "Climate action is necessary to mitigate the impact of global warming on our planet.",
                    "Promoting diversity in all aspects of life enriches our collective experience.",
                    "Social justice begins with acknowledging and addressing systemic issues.",
                    "Collaboration and cooperation are vital for overcoming societal challenges.",
                    "Building strong communities requires active engagement and participation.",
                    "Ensuring access to quality healthcare is a fundamental human right.",
                    "Civic engagement is a responsibility we all share for a thriving society.",
                    "Addressing poverty and economic inequality is key to creating a fair society.",
                    "Promoting tolerance and understanding fosters harmony in diverse societies.",
                    "Investing in renewable energy sources is crucial for a sustainable future.",
                    "Advocating for human rights is a duty that transcends cultural boundaries.",
                    "Ensuring equal access to education empowers individuals and uplifts communities.",
                    "Building bridges of communication fosters unity in a diverse world.",
                    "Sustainable urban planning is essential for creating livable and resilient cities.",
                    "Championing gender equality benefits society as a whole.",
                    "Fostering a culture of empathy can lead to positive societal transformation.",
                    "Preserving cultural heritage is important for maintaining a rich tapestry of identities.",
                    "Supporting initiatives that combat discrimination contributes to a fairer society.",
                    "Protecting the rights of marginalized groups is a cornerstone of social progress.",
                    "Investing in technology for social good can address pressing societal challenges.",
                    "Promoting ethical business practices contributes to a sustainable and just economy.",
                    "Access to clean water and sanitation is a fundamental human right.",
                    "Championing youth empowerment is an investment in the future of society.",
                    "Building resilient communities helps mitigate the impact of natural disasters.",
                    "Advocating for  rights promotes inclusivity and acceptance.",
                    "Ensuring food security is essential for the well-being of communities worldwide.",
                    "Promoting responsible consumption and production is key to sustainability.",
                    "Supporting initiatives that combat human trafficking is crucial for global justice.",
                    "Advocating for affordable housing helps address homelessness and inequality.",
                    "Promoting transparency and accountability strengthens democratic institutions.",
                    "Investing in education for all can break the cycle of poverty and ignorance.",
                    "Addressing the digital divide is crucial for equitable access to information.",
                    "Championing policies that promote social cohesion benefits society at large.",
                    "Preserving biodiversity is essential for the health of our planet.",
                    "Fostering a sense of community builds resilience in times of crisis.",
                    "Ensuring fair and ethical labor practices contributes to a just society.",
                    "Supporting refugee and migrant rights is an expression of solidarity and compassion.",
                    "Promoting mental health awareness reduces stigma and supports well-being.",
                    "Advocating for criminal justice reform promotes fairness and equality.",
                    "Investing in arts and culture enriches the fabric of society.",
                    "Championing disability rights ensures inclusivity and accessibility for all.",
                    "Addressing cybersecurity challenges is crucial for protecting digital societies.",
                    "Promoting peace-building initiatives contributes to global stability.",
                    "Ensuring access to quality education for girls empowers future generations.",
                    "Supporting community-based healthcare initiatives improves overall well-being.",
                    "Advocating for sustainable agriculture practices is essential for food security.",
                    "Championing animal rights promotes compassion and ethical treatment.",
                    "Preserving indigenous rights is important for cultural diversity and heritage.",
                ]

                synapse.completion = response
                if "i'm sorry" in response.lower():
                    new_response = random.choice(alternative_responses_social_issues)
                    response = chain.invoke(
                        {"role": role, "input": new_response}
                    )
                    synapse.completion = new_response

                synapse_latency = time.time() - t0

                if self.config.wandb.on:
                    self.log_event(
                        timing=synapse_latency,
                        prompt=message,
                        completion=response,
                        system_prompt=self.system_prompt,
                        extra_info=self.get_cost_logging(cb)
                    )

            bt.logging.debug(f"✅ Served Response: {response}")
            self.step += 1

            return synapse
        except Exception as e:
            bt.logging.error(f"Error in forward: {e}")
            synapse.completion = "Error: " + str(e)
        finally:
            if self.config.neuron.stop_on_forward_exception:
                self.should_exit = True
            return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    with OpenAIMiner() as miner:
        while True:
            miner.log_status()
            time.sleep(5)

            if miner.should_exit:
                bt.logging.warning("Ending miner...")
                break
