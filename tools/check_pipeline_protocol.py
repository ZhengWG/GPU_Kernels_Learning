"""Abstract buffer ownership checks; NOT a CUDA/PTX memory-model simulator."""
import random


class Slot:
    def __init__(self, consumers=2):
        self.consumers = consumers
        self.epoch = 0  # Full generation number, unlike a hardware parity bit.
        self.state = 'empty'
        self.started, self.finished, self.released = set(), set(), set()

    def require(self, condition):
        if not condition:
            raise ValueError('invalid buffer ownership or generation')

    def load(self, epoch):
        self.require(epoch == self.epoch and self.state == 'empty')
        self.state = 'loading'

    def loaded(self, epoch):
        self.require(epoch == self.epoch and self.state == 'loading')
        self.state = 'ready'

    def compute(self, epoch, consumer):
        self.require(epoch == self.epoch and self.state in ('ready','computing'))
        self.require(0 <= consumer < self.consumers and consumer not in self.started)
        self.started.add(consumer)
        self.state = 'computing'

    def completed(self, epoch, consumer):
        self.require(epoch == self.epoch and consumer in self.started and consumer not in self.finished)
        self.finished.add(consumer)

    def release(self, epoch, consumer):
        self.require(epoch == self.epoch and consumer in self.finished and consumer not in self.released)
        self.released.add(consumer)
        if len(self.released) == self.consumers:
            self.epoch += 1
            self.state = 'empty'
            self.started.clear()
            self.finished.clear()
            self.released.clear()

    def drain(self):
        self.require(self.state == 'empty')


def allocation_columns(required):
    if required <= 0 or required > 512:
        raise ValueError('SM100 worksheet demand outside 1..512 columns')
    result = 32
    while result < required:
        result *= 2
    return result


def rejected(operation):
    try:
        operation()
    except ValueError:
        return
    raise AssertionError('unsafe sequence was accepted')


def main():
    runs = 0
    for stages in [1,2,3]:
        for seed in range(12):
            rng = random.Random(seed)
            slots = [Slot() for _ in range(stages)]
            pending = {}
            completed = []
            next_tile = 0
            while next_tile < 11 or pending:
                # Produce as far ahead as each slot's ownership permits.
                if next_tile < 11 and slots[next_tile%stages].state == 'empty':
                    slot = slots[next_tile%stages]
                    slot.load(next_tile//stages)
                    pending[next_tile] = ('loading',set(),set(),set())
                    next_tile += 1
                tile = rng.choice(list(pending))
                slot,epoch = slots[tile%stages],tile//stages
                state,started,finished,released = pending[tile]
                if state == 'loading':
                    slot.loaded(epoch)
                    pending[tile] = ('computing',started,finished,released)
                else:
                    actions = [('start',c) for c in range(2) if c not in started]
                    actions += [('finish',c) for c in started if c not in finished]
                    actions += [('release',c) for c in finished if c not in released]
                    action,c = rng.choice(actions)
                    if action == 'start':
                        slot.compute(epoch,c)
                        started.add(c)
                    elif action == 'finish':
                        slot.completed(epoch,c)
                        finished.add(c)
                    else:
                        slot.release(epoch,c)
                        released.add(c)
                    if len(released) == 2:
                        completed.append(tile)
                        del pending[tile]
            for slot in slots:
                slot.drain()
            assert sorted(completed) == list(range(11))
            runs += 1
    slot = Slot()
    rejected(lambda:slot.compute(0,0))
    slot.load(0)
    rejected(lambda:slot.compute(0,0))
    rejected(lambda:slot.load(0))
    rejected(slot.drain)
    slot.loaded(0)
    slot.compute(0,0)
    rejected(lambda:slot.release(0,0))
    slot.completed(0,0)
    slot.release(0,0)
    rejected(lambda:slot.load(1))  # Consumer 1 has not finished.
    slot.compute(0,1)
    slot.completed(0,1)
    slot.release(0,1)
    slot.load(1)
    rejected(lambda:slot.loaded(0))  # Late notification from prior generation.
    slot.loaded(1)
    rejected(lambda:slot.compute(0,0))
    assert [allocation_columns(x) for x in [1,32,33,128,256,384,512]] == [32,32,64,128,256,512,512]
    rejected(lambda:allocation_columns(513))
    print(f'Pipeline: {runs} interleavings passed; 8 unsafe transitions rejected.')
    print('TMEM worksheet: 7 sizes passed and oversized demand rejected.')
    print('Abstract ownership only; no hardware phase, visibility or instruction validation.')


if __name__ == '__main__':
    main()
